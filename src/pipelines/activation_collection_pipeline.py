from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.models.flux_model import FluxModel
from src.shift.manager import ShiftInterventionManager


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
    ) -> None:
        self.model = model
        self.intervention_manager = intervention_manager
        self.dataset = dataset
        self.generation_config = generation_config
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.logger = logger

    def collect(self) -> dict:
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

        for pair_index, pair in enumerate(
            self.dataset
        ):
            pair_seed = self.seed + pair_index

            self.logger.info(
                "Collecting pair=%s, seed=%d",
                pair.name,
                pair_seed,
            )

            run_specs = [
                (
                    "negative",
                    pair.negative_prompt,
                ),
                (
                    "positive",
                    pair.positive_prompt,
                ),
            ]

            for prompt_role, prompt in run_specs:
                self.intervention_manager.begin_prompt_run(
                    pair_name=pair.name,
                    prompt_role=prompt_role,
                )

                self.logger.info(
                    "Pair=%s, role=%s",
                    pair.name,
                    prompt_role,
                )
                self.logger.info(
                    "Prompt: %s",
                    prompt,
                )

                # The same seed is recreated for positive
                # and negative prompts within the pair.
                generator = torch.Generator(
                    device="cpu"
                ).manual_seed(pair_seed)

                with torch.inference_mode():
                    _ = pipe(
                        prompt=prompt,
                        width=int(
                            self.generation_config.width
                        ),
                        height=int(
                            self.generation_config.height
                        ),
                        num_inference_steps=int(
                            self.generation_config
                            .num_inference_steps
                        ),
                        guidance_scale=float(
                            self.generation_config
                            .guidance_scale
                        ),
                        max_sequence_length=int(
                            self.generation_config
                            .max_sequence_length
                        ),
                        num_images_per_prompt=int(
                            self.generation_config
                            .num_images_per_prompt
                        ),
                        generator=generator,
                        output_type="pil",
                    )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        saved_metadata_path = (
            self.intervention_manager
            .save_collected_activations()
        )

        report = (
            self.model
            .get_intervention_report()
        )

        report_path = (
            self.output_dir
            / "collection_report.yaml"
        )

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
            "activation_metadata_path": (
                saved_metadata_path
            ),
        }
