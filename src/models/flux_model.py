from __future__ import annotations

from typing import Any

import torch
from diffusers import FluxPipeline
from omegaconf import DictConfig, OmegaConf


DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class FluxModel:
    """
    Wrapper around diffusers.FluxPipeline.

    Intervention hooks are installed before Accelerate configures
    sequential CPU offloading.
    """

    def __init__(
        self,
        repo_id: str,
        dtype: str,
        memory: DictConfig | dict[str, Any],
        load: DictConfig | dict[str, Any],
        device: torch.device,
        intervention_manager: Any | None = None,
    ) -> None:
        self.repo_id = str(repo_id)
        self.device = device
        self.dtype = self._resolve_dtype(str(dtype))

        self.memory_config = self._to_dict(memory)
        self.load_config = self._to_dict(load)

        self.intervention_manager = (
            intervention_manager
        )

        self._pipeline: FluxPipeline | None = None

    @staticmethod
    def _to_dict(
        config: DictConfig | dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(config, DictConfig):
            value = OmegaConf.to_container(
                config,
                resolve=True,
            )

            if not isinstance(value, dict):
                raise TypeError(
                    "Expected a mapping configuration."
                )

            return value

        return dict(config)

    @staticmethod
    def _resolve_dtype(
        dtype_name: str,
    ) -> torch.dtype:
        try:
            return DTYPES[dtype_name]
        except KeyError as error:
            raise ValueError(
                f"Unsupported dtype: {dtype_name!r}. "
                f"Available values: {list(DTYPES)}"
            ) from error

    def prepare_for_inference(self) -> None:
        if self._pipeline is not None:
            return

        pipeline = self._load_pipeline()

        # This must happen before sequential CPU offload hooks
        # are installed by Accelerate.
        self._install_interventions(pipeline)

        self._configure_memory(pipeline)

        if bool(
            self.memory_config.get(
                "enable_vae_slicing",
                False,
            )
        ):
            pipeline.enable_vae_slicing()

        if bool(
            self.memory_config.get(
                "enable_vae_tiling",
                False,
            )
        ):
            pipeline.enable_vae_tiling()

        self._pipeline = pipeline

    def _load_pipeline(self) -> FluxPipeline:
        load_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "use_safetensors": bool(
                self.load_config.get(
                    "use_safetensors",
                    True,
                )
            ),
            "local_files_only": bool(
                self.load_config.get(
                    "local_files_only",
                    False,
                )
            ),
        }

        return FluxPipeline.from_pretrained(
            self.repo_id,
            **load_kwargs,
        )

    def _install_interventions(
        self,
        pipeline: FluxPipeline,
    ) -> None:
        if self.intervention_manager is None:
            return

        self.intervention_manager.install(
            pipeline.transformer,
        )

    def _configure_memory(
        self,
        pipeline: FluxPipeline,
    ) -> None:
        strategy = str(
            self.memory_config.get(
                "strategy",
                "sequential_cpu_offload",
            )
        )

        if strategy == "model_cpu_offload":
            pipeline.enable_model_cpu_offload()

        elif strategy == "sequential_cpu_offload":
            pipeline.enable_sequential_cpu_offload()

        elif strategy == "cuda":
            if self.device.type != "cuda":
                raise ValueError(
                    "memory.strategy=cuda requires "
                    "device=cuda."
                )

            pipeline.to(self.device)

        else:
            raise ValueError(
                f"Unknown memory strategy: {strategy!r}"
            )

    def get_pipeline(self) -> FluxPipeline:
        if self._pipeline is None:
            raise RuntimeError(
                "FLUX is not initialized. "
                "Call prepare_for_inference() first."
            )

        return self._pipeline

    def get_intervention_report(
        self,
    ) -> dict[str, Any]:
        if self.intervention_manager is None:
            return {
                "type": "none",
                "installed": False,
            }

        return self.intervention_manager.report()

    def remove_interventions(self) -> None:
        if self.intervention_manager is not None:
            self.intervention_manager.remove()

    def encode_prompt(
        self,
        prompt: str,
        max_sequence_length: int,
        num_images_per_prompt: int = 1,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Encode a prompt with the original FLUX text encoders.

        Returns:
            prompt_embeds:
                T5 sequence embeddings.

            pooled_prompt_embeds:
                CLIP pooled embeddings.

            text_ids:
                FLUX text position identifiers.
        """
        pipe = self.get_pipeline()

        with torch.inference_mode():
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=(
                    num_images_per_prompt
                ),
                max_sequence_length=(
                    max_sequence_length
                ),
            )

        return (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        )


    def encode_pooled_prompt(
        self,
        prompt: str,
        max_sequence_length: int,
    ) -> torch.Tensor:
        """
        Return only the pooled CLIP embedding.

        The full public encode_prompt API is used rather
        than depending on a private Diffusers method.
        """
        (
            _,
            pooled_prompt_embeds,
            _,
        ) = self.encode_prompt(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=1,
        )

        return pooled_prompt_embeds
