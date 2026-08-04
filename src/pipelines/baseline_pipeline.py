from __future__ import annotations

import importlib.metadata
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.models.flux_model import FluxModel


class BaselineInferencePipeline:
    """
    Generate images with the original, unmodified FLUX pipeline.
    """

    def __init__(
        self,
        model: FluxModel,
        prompt: str,
        seed: int,
        generation_config: DictConfig,
        output_config: DictConfig,
        logger: logging.Logger,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.seed = seed
        self.generation_config = generation_config
        self.output_config = output_config
        self.logger = logger

    def generate(self) -> list[Path]:
        pipe = self.model.get_pipeline()

        generator = torch.Generator(
            device="cpu",
        ).manual_seed(self.seed)

        generation_kwargs: dict[str, Any] = {
            "prompt": self.prompt,
            "width": int(self.generation_config.width),
            "height": int(self.generation_config.height),
            "num_inference_steps": int(self.generation_config.num_inference_steps),
            "guidance_scale": float(self.generation_config.guidance_scale),
            "max_sequence_length": int(self.generation_config.max_sequence_length),
            "num_images_per_prompt": int(self.generation_config.num_images_per_prompt),
            "generator": generator,
            "output_type": "pil",
        }

        self.logger.info(
            "Generating with parameters:\n%s",
            OmegaConf.to_yaml(
                self.generation_config,
                resolve=True,
            ),
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            result = pipe(**generation_kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        output_paths = self._save_images(
            images=result.images,
        )

        self._save_run_metadata(
            output_paths=output_paths,
        )

        return output_paths

    def _save_images(
        self,
        images: list[Any],
    ) -> list[Path]:
        output_directory = Path(str(self.output_config.directory))
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        configured_filename = Path(str(self.output_config.filename))

        output_paths: list[Path] = []

        for index, image in enumerate(images):
            if len(images) == 1:
                filename = configured_filename
            else:
                filename = configured_filename.parent / (
                    f"{configured_filename.stem}" f"_{index:03d}" f"{configured_filename.suffix}"
                )

            output_path = output_directory / filename

            image.save(output_path)
            output_paths.append(output_path)

        return output_paths

    def _save_run_metadata(
        self,
        output_paths: list[Path],
    ) -> None:
        output_directory = Path(str(self.output_config.directory))

        metadata: dict[str, Any] = {
            "model": {
                "repo_id": self.model.repo_id,
                "dtype": str(self.model.dtype),
                "memory": self.model.memory_config,
                "load": self.model.load_config,
            },
            "intervention": (self.model.get_intervention_report()),
            "prompt": self.prompt,
            "seed": self.seed,
            "generation": OmegaConf.to_container(
                self.generation_config,
                resolve=True,
            ),
            "output_files": [str(path.name) for path in output_paths],
            "environment": {
                "python_packages": {
                    "torch": self._package_version("torch"),
                    "diffusers": self._package_version("diffusers"),
                    "transformers": self._package_version("transformers"),
                    "accelerate": self._package_version("accelerate"),
                    "hydra-core": self._package_version("hydra-core"),
                },
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            },
        }

        if torch.cuda.is_available():
            metadata["environment"]["gpu"] = torch.cuda.get_device_name(0)
            metadata["environment"]["peak_memory_allocated_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024**3,
                3,
            )
            metadata["environment"]["peak_memory_reserved_gb"] = round(
                torch.cuda.max_memory_reserved() / 1024**3,
                3,
            )

        metadata_path = output_directory / "run_metadata.yaml"

        OmegaConf.save(
            config=OmegaConf.create(metadata),
            f=metadata_path,
        )

        self.logger.info(
            "Saved run metadata: %s",
            metadata_path,
        )

    @staticmethod
    def _package_version(
        package_name: str,
    ) -> str | None:
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return None
