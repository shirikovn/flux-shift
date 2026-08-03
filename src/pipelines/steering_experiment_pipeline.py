from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
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
from src.utils.run_output_store import (
    RunOutputStore,
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
        resume_config: Any,
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

        self.generation_dict = (
            OmegaConf.to_container(
                self.generation_config,
                resolve=True,
            )
        )

        if not isinstance(
            self.generation_dict,
            dict,
        ):
            raise TypeError(
                "generation_config must resolve to a mapping."
            )

        model_report = self.model.get_model_report()

        self.model_identity = {
            "repo_id": model_report.get("repo_id"),
            "revision": model_report.get("revision"),
            "dtype": model_report.get("dtype"),
            "pipeline_class": (
                model_report.get("pipeline_class")
            ),
            "transformer_class": (
                model_report.get("transformer_class")
            ),
        }

        self.output_store = RunOutputStore(
            output_dir=self.output_dir,
            config=resume_config,
            logger=self.logger,
        )

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

        metadata_path = (
            self.output_dir
            / "experiment_metadata.yaml"
        )

        self._save_experiment_metadata(
            run_records=run_records,
            status="running",
            metadata_path=metadata_path,
        )

        try:
            for case in self.cases:
                case_name = str(case["name"])
                prompt = str(case["prompt"])
                operation = str(
                    case["operation"]
                )

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

                self._save_experiment_metadata(
                    run_records=run_records,
                    status="running",
                    metadata_path=metadata_path,
                )

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
                            use_classifier=(
                                use_classifier
                            ),
                            use_pooled=use_pooled,
                            pooled_strength=(
                                pooled_strength
                            ),
                            pooled_similarity_mode=(
                                pooled_similarity_mode
                            ),
                        )

                        run_records.append(record)

                        self._save_experiment_metadata(
                            run_records=run_records,
                            status="running",
                            metadata_path=metadata_path,
                        )

        except BaseException:
            self._save_experiment_metadata(
                run_records=run_records,
                status="interrupted",
                metadata_path=metadata_path,
            )
            raise

        self._save_experiment_metadata(
            run_records=run_records,
            status="completed",
            metadata_path=metadata_path,
        )

        summary = self._summarize_runs(
            run_records
        )

        self.logger.info(
            "Saved metadata: %s",
            metadata_path,
        )

        self.logger.info(
            "Run summary: %s",
            summary,
        )

        return {
            "output_dir": str(
                self.output_dir
            ),
            "metadata_path": str(
                metadata_path
            ),
            "num_runs": len(run_records),
            "summary": summary,
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
                f"__gamma_"
                f"{self._format_number(strength)}"
            )
        )

        run_name = (
            f"{case_name}__{suffix}"
        )

        specification = self._build_run_specification(
            case_name=case_name,
            prompt=prompt,
            operation=operation,
            strength=strength,
            schedule_name=schedule_name,
            blocks=blocks,
            steps=steps,
            use_classifier=use_classifier,
            use_pooled=use_pooled,
            pooled_strength=pooled_strength,
            pooled_similarity_mode=(
                pooled_similarity_mode
            ),
        )

        specification_hash = (
            RunOutputStore.hash_specification(
                specification
            )
        )

        run_id = specification_hash[:16]

        filename = (
            f"{self._sanitize(case_name)}"
            f"__{self._sanitize(suffix)}"
            f"__{run_id}.png"
        )

        paths = self.output_store.build_paths(
            run_id=run_id,
            filename=filename,
        )

        existing_record, resume_action = (
            self.output_store.prepare(
                paths=paths,
                run_id=run_id,
                specification_hash=(
                    specification_hash
                ),
                specification=specification,
            )
        )

        if existing_record is not None:
            return existing_record

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
            "Run=%s, run_id=%s, operation=%s, "
            "strength=%g, blocks=%s, steps=%s, "
            "resume_action=%s",
            run_name,
            run_id,
            operation,
            strength,
            blocks,
            steps,
            resume_action,
        )

        generation_started = time.perf_counter()

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

        try:
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
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        generation_seconds = (
            time.perf_counter()
            - generation_started
        )

        statistics = (
            self.intervention_manager
            .steering_statistics()
        )

        pooled_statistics = (
            self.intervention_manager
            .pooled_statistics()
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

        record = {
            "schema_version": 1,
            "status": "saving",
            "run_id": run_id,
            "run_name": run_name,
            "specification_hash": (
                specification_hash
            ),
            "specification": specification,
            "resume_action": resume_action,
            "case_name": case_name,
            "prompt": prompt,
            "seed": self.seed,
            "operation": operation,
            "base_strength": strength,
            "use_classifier": use_classifier,
            "schedule": schedule_name,
            "blocks": blocks,
            "steps": steps,
            "generation_seconds": (
                generation_seconds
            ),
            "completed_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "steering_statistics": statistics,
            "pooled": {
                "enabled": use_pooled,
                "strength": pooled_strength,
                "similarity_mode": (
                    pooled_similarity_mode
                ),
                "statistics": pooled_statistics,
            },
        }

        completed_record = (
            self.output_store.save_completed(
                image=result.images[0],
                record=record,
                paths=paths,
            )
        )

        self.logger.info(
            "Saved completed run: %s",
            paths.image_path,
        )

        return completed_record

    def _build_run_specification(
        self,
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
        return {
            "schema_version": 1,
            "model": self.model_identity,
            "generation": self.generation_dict,
            "seed": self.seed,
            "case": {
                "name": case_name,
                "prompt": prompt,
                "operation": operation,
            },
            "steering": {
                "schedule": schedule_name,
                "strength": strength,
                "blocks": blocks,
                "steps": steps,
                "use_classifier": (
                    use_classifier
                ),
                "pooled": {
                    "enabled": use_pooled,
                    "strength": pooled_strength,
                    "similarity_mode": (
                        pooled_similarity_mode
                    ),
                },
            },
        }


    def _save_experiment_metadata(
        self,
        run_records: list[dict[str, Any]],
        status: str,
        metadata_path: Path,
    ) -> None:
        metadata = {
            "schema_version": 1,
            "status": status,
            "seed": self.seed,
            "model": self.model_identity,
            "generation": self.generation_dict,
            "default_strengths": (
                self.strengths
            ),
            "cases": self.cases,
            "schedules": self.schedules,
            "summary": self._summarize_runs(
                run_records
            ),
            "runs": run_records,
            "intervention": (
                self.model
                .get_intervention_report()
            ),
        }

        self.output_store.atomic_save_yaml(
            data=metadata,
            path=metadata_path,
        )


    @staticmethod
    def _summarize_runs(
        run_records: list[dict[str, Any]],
    ) -> dict[str, int]:
        actions = [
            str(
                record.get(
                    "resume_action",
                    "generated",
                )
            )
            for record in run_records
        ]

        return {
            "total": len(run_records),
            "generated": actions.count(
                "generated"
            ),
            "skipped_existing": actions.count(
                "skipped_existing"
            ),
            "repaired": actions.count(
                "repaired"
            ),
            "overwritten": actions.count(
                "overwritten"
            ),
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
