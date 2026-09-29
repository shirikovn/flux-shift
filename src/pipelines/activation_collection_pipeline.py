from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig, OmegaConf

from src.models.flux_model import FluxModel
from src.shift.manager import ShiftInterventionManager

StepEndCallback = Callable[
    [Any, int, Any, dict[str, Any]],
    dict[str, Any],
]


class ActivationCollectionPipeline:
    def __init__(
        self,
        model: FluxModel,
        intervention_manager: ShiftInterventionManager,
        dataset,
        generation_config: DictConfig,
        output_dir: str,
        seed: int,
        logger: logging.Logger,
        seed_stride: int = 1,
        seeds_per_pair: int = 1,
        replica_seed_stride: int = 1,
    ) -> None:
        self.model = model
        self.intervention_manager = intervention_manager
        self.dataset = dataset
        self.generation_config = generation_config
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.seed_stride = int(seed_stride)
        self.seeds_per_pair = int(seeds_per_pair)
        self.replica_seed_stride = int(replica_seed_stride)
        self.logger = logger

        if self.seed_stride <= 0:
            raise ValueError("seed_stride must be positive.")
        if self.seeds_per_pair <= 0:
            raise ValueError("seeds_per_pair must be positive.")
        if self.replica_seed_stride <= 0:
            raise ValueError("replica_seed_stride must be positive.")
        if (
            self.seeds_per_pair > 1
            and (self.seeds_per_pair - 1) * self.replica_seed_stride
            >= self.seed_stride
        ):
            raise ValueError(
                "Replica seeds overlap the next prompt pair. Require "
                "(seeds_per_pair - 1) * replica_seed_stride < seed_stride."
            )

    def collect(self) -> dict[str, Any]:
        pipe = self.model.get_pipeline()

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.logger.info(
            "Collection generation config:\n%s",
            OmegaConf.to_yaml(
                self.generation_config,
                resolve=True,
            ),
        )

        stop_after_step = self._resolve_stop_after_step()
        stop_callback = self._make_stop_callback(stop_after_step)

        num_inference_steps = int(self.generation_config.num_inference_steps)

        active_steps = self.intervention_manager.state.active_steps

        requested_steps = sorted(active_steps) if active_steps is not None else None

        executed_steps = stop_after_step + 1 if stop_after_step is not None else num_inference_steps

        self.logger.info(
            "Fast activation collection enabled: "
            "requested_steps=%s, "
            "transformer_steps_per_prompt=%d/%d, "
            "vae_decode=false",
            requested_steps,
            executed_steps,
            num_inference_steps,
        )

        for pair_index, pair in enumerate(self.dataset):
            for replica_index in range(self.seeds_per_pair):
                pair_seed = (
                    self.seed
                    + pair_index * self.seed_stride
                    + replica_index * self.replica_seed_stride
                )
                collection_pair_name = (
                    pair.name
                    if self.seeds_per_pair == 1
                    else f"{pair.name}__replica_{replica_index:02d}"
                )

                self.logger.info(
                    "Collecting pair=%s, source_pair=%s, replica=%d/%d, seed=%d",
                    collection_pair_name,
                    pair.name,
                    replica_index + 1,
                    self.seeds_per_pair,
                    pair_seed,
                )

                run_specs = [
                    ("negative", pair.negative_prompt),
                    ("positive", pair.positive_prompt),
                ]

                for prompt_role, prompt in run_specs:
                    self.intervention_manager.begin_prompt_run(
                        pair_name=collection_pair_name,
                        prompt_role=prompt_role,
                    )

                    self.logger.info(
                        "Pair=%s, role=%s",
                        collection_pair_name,
                        prompt_role,
                    )
                    self.logger.info("Prompt: %s", prompt)

                    # Recreate the generator so the two members of each
                    # counterfactual pair use exactly the same noise.
                    generator = torch.Generator(device="cpu").manual_seed(
                        pair_seed
                    )

                    pipeline_kwargs: dict[str, Any] = {
                        "prompt": prompt,
                        "width": int(self.generation_config.width),
                        "height": int(self.generation_config.height),
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": float(
                            self.generation_config.guidance_scale
                        ),
                        "max_sequence_length": int(
                            self.generation_config.max_sequence_length
                        ),
                        "num_images_per_prompt": int(
                            self.generation_config.num_images_per_prompt
                        ),
                        "generator": generator,
                        # No decoded image is required for activation
                        # collection, so skip the VAE.
                        "output_type": "latent",
                    }

                    if stop_callback is not None:
                        pipeline_kwargs.update(
                            {
                                "callback_on_step_end": stop_callback,
                                "callback_on_step_end_tensor_inputs": [],
                            }
                        )

                    with torch.inference_mode():
                        _ = pipe(**pipeline_kwargs)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        saved_metadata_path = self.intervention_manager.save_collected_activations()

        report = self.model.get_intervention_report()

        report["collection_execution"] = {
            "requested_steps": requested_steps,
            "configured_inference_steps": (num_inference_steps),
            "stop_after_step": stop_after_step,
            "executed_transformer_steps_per_prompt": (executed_steps),
            "source_prompt_pairs": len(self.dataset),
            "seeds_per_pair": self.seeds_per_pair,
            "replica_seed_stride": self.replica_seed_stride,
            "effective_prompt_pairs": len(self.dataset) * self.seeds_per_pair,
            "output_type": "latent",
            "vae_decode": False,
            "early_stop_enabled": (
                stop_after_step is not None and stop_after_step < num_inference_steps - 1
            ),
        }

        report_path = self.output_dir / "collection_report.yaml"

        OmegaConf.save(
            config=OmegaConf.create(report),
            f=report_path,
        )

        self.logger.info(
            "Saved collection report: %s",
            report_path,
        )
        self.logger.info(
            "Saved vector metadata: %s",
            saved_metadata_path,
        )

        return {
            "report_path": str(report_path),
            "activation_metadata_path": (saved_metadata_path),
        }

    def _resolve_stop_after_step(
        self,
    ) -> int | None:
        """
        Return the final transformer step needed by the
        activation collector.

        active_steps=None means that every configured diffusion
        step should be executed.
        """
        active_steps = self.intervention_manager.state.active_steps

        if active_steps is None:
            return None

        if not active_steps:
            raise ValueError(
                "Activation collection requires at least " "one active diffusion step."
            )

        num_inference_steps = int(self.generation_config.num_inference_steps)

        invalid_steps = sorted(
            step for step in active_steps if step < 0 or step >= num_inference_steps
        )

        if invalid_steps:
            raise ValueError(
                "Collection steps are outside the configured "
                "diffusion schedule. "
                f"Invalid steps: {invalid_steps}; "
                f"num_inference_steps={num_inference_steps}."
            )

        return max(active_steps)

    @staticmethod
    def _make_stop_callback(
        stop_after_step: int | None,
    ) -> StepEndCallback | None:
        if stop_after_step is None:
            return None

        def stop_callback(
            pipeline: Any,
            step_index: int,
            timestep: Any,
            callback_kwargs: dict[str, Any],
        ) -> dict[str, Any]:
            del timestep

            if step_index >= stop_after_step:
                if not hasattr(
                    pipeline,
                    "_interrupt",
                ):
                    raise RuntimeError(
                        "The installed Diffusers pipeline "
                        "does not expose the interruption "
                        "mechanism required for fast "
                        "activation collection."
                    )

                # FluxPipeline checks this flag at the start
                # of every remaining denoising iteration.
                setattr(
                    pipeline,
                    "_interrupt",
                    True,
                )

            return callback_kwargs

        return stop_callback
