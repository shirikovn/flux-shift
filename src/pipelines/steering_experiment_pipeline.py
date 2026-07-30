from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch
from omegaconf import (
    DictConfig,
    ListConfig,
    OmegaConf,
)

from src.models.flux_model import FluxModel
from src.shift.manager import (
    ShiftInterventionManager,
)


class SteeringExperimentPipeline:
    """
    Runs block-range and strength ablations.

    Baseline is generated once per case. Every steered run
    reuses exactly the same seed and initial noise.
    """

    def __init__(
        self,
        model: FluxModel,
        intervention_manager: ShiftInterventionManager,
        cases: Any,
        schedules: Any,
        strengths: Any,
        generation_config: DictConfig,
        output_dir: str,
        seed: int,
        logger: logging.Logger,
    ) -> None:
        self.model = model
        self.intervention_manager = (
            intervention_manager
        )
        self.generation_config = (
            generation_config
        )
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.logger = logger

        self.cases = self._as_list(
            cases,
            name="cases",
        )
        self.schedules = self._as_list(
            schedules,
            name="schedules",
        )

        self.strengths = [
            float(value)
            for value in self._to_container(
                strengths
            )
        ]

        if not self.strengths:
            raise ValueError(
                "At least one steering strength "
                "must be configured."
            )

        self._validate_schedules()

    @staticmethod
    def _to_container(
        value: Any,
    ) -> Any:
        if isinstance(
            value,
            (DictConfig, ListConfig),
        ):
            return OmegaConf.to_container(
                value,
                resolve=True,
            )

        return value

    def _as_list(
        self,
        value: Any,
        name: str,
    ) -> list[dict[str, Any]]:
        result = self._to_container(value)

        if not isinstance(result, list):
            raise TypeError(
                f"experiment.{name} must be a list."
            )

        return result

    def _validate_schedules(self) -> None:
        for schedule in self.schedules:
            if "name" not in schedule:
                raise ValueError(
                    "Every schedule requires a name."
                )

            blocks = schedule.get("blocks")
            steps = schedule.get("steps")

            if not isinstance(blocks, list):
                raise TypeError(
                    f"Schedule {schedule['name']!r} "
                    "requires a blocks list."
                )

            if not isinstance(steps, list):
                raise TypeError(
                    f"Schedule {schedule['name']!r} "
                    "requires a steps list."
                )

            if not blocks:
                raise ValueError(
                    f"Schedule {schedule['name']!r} "
                    "has no blocks."
                )

            if not steps:
                raise ValueError(
                    f"Schedule {schedule['name']!r} "
                    "has no steps."
                )

    def run(self) -> dict[str, Any]:
        pipe = self.model.get_pipeline()

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        run_records: list[
            dict[str, Any]
        ] = []

        for case in self.cases:
            case_name = str(case["name"])
            prompt = str(case["prompt"])
            operation = str(case["operation"])

            baseline = self._generate_one(
                pipe=pipe,
                case_name=case_name,
                prompt=prompt,
                operation=operation,
                strength=0.0,
                schedule_name="baseline",
                blocks=[],
                steps=[],
                use_classifier=False,
                use_pooled=False,
                pooled_strength=0.0,
                pooled_similarity_mode="raw",
            )
            run_records.append(baseline)

            for schedule in self.schedules:
                schedule_name = str(
                    schedule["name"]
                )

                blocks = [
                    int(value)
                    for value
                    in schedule["blocks"]
                ]

                steps = [
                    int(value)
                    for value
                    in schedule["steps"]
                ]

                schedule_strengths = (
                    schedule.get(
                        "strengths",
                        self.strengths,
                    )
                )

                use_classifier = bool(
                    schedule.get(
                        "use_classifier",
                        False,
                    )
                )

                use_pooled = bool(
                    schedule.get(
                        "use_pooled",
                        False,
                    )
                )

                pooled_strength = float(
                    schedule.get(
                        "pooled_strength",
                        0.0,
                    )
                )

                pooled_similarity_mode = str(
                    schedule.get(
                        "pooled_similarity_mode",
                        "raw",
                    )
                )

                for raw_strength in (
                    schedule_strengths
                ):
                    strength = float(
                        raw_strength
                    )

                    record = self._generate_one(
                        pipe=pipe,
                        case_name=case_name,
                        prompt=prompt,
                        operation=operation,
                        strength=strength,
                        schedule_name=(
                            schedule_name
                        ),
                        blocks=blocks,
                        steps=steps,
                        use_classifier=use_classifier,
                        use_pooled=use_pooled,
                        pooled_strength=pooled_strength,
                        pooled_similarity_mode=pooled_similarity_mode,
                    )

                    run_records.append(record)

        metadata = {
            "seed": self.seed,
            "generation": (
                OmegaConf.to_container(
                    self.generation_config,
                    resolve=True,
                )
            ),
            "default_strengths": (
                self.strengths
            ),
            "cases": self.cases,
            "schedules": self.schedules,
            "runs": run_records,
            "intervention": (
                self.model
                .get_intervention_report()
            ),
        }

        metadata_path = (
            self.output_dir
            / "experiment_metadata.yaml"
        )

        OmegaConf.save(
            config=OmegaConf.create(
                metadata
            ),
            f=metadata_path,
        )

        self.logger.info(
            "Saved metadata: %s",
            metadata_path,
        )

        return {
            "output_dir": str(
                self.output_dir
            ),
            "metadata_path": str(
                metadata_path
            ),
            "num_runs": len(run_records),
        }

    def _generate_one(
        self,
        pipe: Any,
        case_name: str,
        prompt: str,
        operation: str,
        strength: float,
        schedule_name: str,
        blocks: list[int],
        steps: list[int],
        use_classifier: bool,
        use_pooled: bool,
        pooled_strength: float,
        pooled_similarity_mode: str,
    ) -> dict[str, Any]:
        is_baseline = (
            schedule_name == "baseline"
        )

        regularization_name = (
            "svm"
            if use_classifier
            else "static"
        )
        
        suffix = (
            "baseline"
            if is_baseline
            else (
                f"{schedule_name}"
                f"__{regularization_name}"
                f"__{operation}"
                f"__gamma_{self._format_number(strength)}"
            )
        )

        run_name = (
            f"{case_name}__{suffix}"
        )

        self.intervention_manager.configure_locations(
            blocks=blocks,
            steps=steps,
        )

        self.intervention_manager.configure_steering(
            operation=operation,
            strength=strength,
            use_classifier=use_classifier,
        )

        self.intervention_manager.reset_steering_statistics()

        self.intervention_manager.configure_pooled_steering(
            enabled=use_pooled,
            operation=operation,
            strength=pooled_strength,
            similarity_mode=(
                pooled_similarity_mode
            ),
        )

        self.intervention_manager.reset_pooled_statistics()

        self.intervention_manager.begin_steering_run(
            run_name=run_name
        )

        generator = torch.Generator(
            device="cpu"
        ).manual_seed(self.seed)

        self.logger.info(
            "Run=%s, operation=%s, strength=%g, "
            "blocks=%s, steps=%s",
            run_name,
            operation,
            strength,
            blocks,
            steps,
        )

        (
            prompt_embeds,
            pooled_prompt_embeds,
            _,
        ) = self.model.encode_prompt(
            prompt=prompt,
            max_sequence_length=int(
                self.generation_config
                .max_sequence_length
            ),
            num_images_per_prompt=1,
        )

        pooled_prompt_embeds = (
            self.intervention_manager
            .apply_pooled_steering(
                pooled_prompt_embeds
            )
        )

        with torch.inference_mode():
            result = pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=(
                    pooled_prompt_embeds
                ),
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
                num_images_per_prompt=1,
                generator=generator,
                output_type="pil",
            )

        filename = (
            f"{self._sanitize(case_name)}"
            f"__{self._sanitize(suffix)}.png"
        )

        image_path = (
            self.output_dir / filename
        )

        result.images[0].save(image_path)

        statistics = (
            self.intervention_manager
            .steering_statistics()
        )

        if statistics is not None:
            self.logger.info(
                "Modified calls: %s; "
                "relative scale mean: %s",
                statistics.get(
                    "modified_calls"
                ),
                statistics.get(
                    "relative_scale_mean"
                ),
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "run_name": run_name,
            "case_name": case_name,
            "prompt": prompt,
            "operation": operation,
            "base_strength": strength,
            "use_classifier": use_classifier,
            "schedule": schedule_name,
            "blocks": blocks,
            "steps": steps,
            "filename": filename,
            "steering_statistics": statistics,
            "pooled": {
                "enabled": use_pooled,
                "strength": pooled_strength,
                "similarity_mode": (
                    pooled_similarity_mode
                ),
                "statistics": (
                    self.intervention_manager
                    .pooled_statistics()
                ),
            },
        }

    @staticmethod
    def _format_number(
        value: float,
    ) -> str:
        return (
            f"{value:g}"
            .replace("-", "m")
            .replace(".", "p")
        )

    @staticmethod
    def _sanitize(
        value: str,
    ) -> str:
        value = value.strip()
        value = re.sub(
            r"\s+",
            "_",
            value,
        )
        value = re.sub(
            r"[^A-Za-z0-9_\-\.]",
            "",
            value,
        )
        return value
