from __future__ import annotations

import csv
import hashlib
import logging
import os
import random

from dataclasses import dataclass
from pathlib import Path

import hydra
import torch

from hydra.utils import instantiate
from omegaconf import (
    DictConfig,
    OmegaConf,
    open_dict,
)

from src.utils.init_utils import set_random_seed
from src.utils.run_manifest import RunManifest


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class I2PCase:
    index: int
    prompt: str
    seed: int

    @property
    def name(self) -> str:
        return f"i2p_{self.index:04d}"


def load_i2p_csv(path: Path) -> list[I2PCase]:
    if not path.is_file():
        raise FileNotFoundError(
            f"I2P CSV does not exist: {path}\n"
            "Run prepare_i2p.py first."
        )

    cases: list[I2PCase] = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "i2p_index",
            "prompt",
            "sd_seed",
        }

        fields = set(reader.fieldnames or [])

        missing = required - fields

        if missing:
            raise RuntimeError(
                f"Missing CSV columns: {sorted(missing)}"
            )

        for row in reader:
            cases.append(
                I2PCase(
                    index=int(row["i2p_index"]),
                    prompt=str(row["prompt"]),
                    seed=int(row["sd_seed"]),
                )
            )

    if not cases:
        raise RuntimeError(
            f"No I2P cases found in {path}"
        )

    indices = [case.index for case in cases]

    if len(indices) != len(set(indices)):
        raise RuntimeError(
            "I2P CSV contains duplicate indices."
        )

    return cases


def select_cases(
    cases: list[I2PCase],
    sample_size: int,
    sample_seed: int,
) -> list[I2PCase]:
    """
    Deterministically shuffle once and take a prefix.

    Consequently:
        sample_size=512
    is a strict prefix of:
        sample_size=1024

    as long as sample_seed stays unchanged.
    """

    selected = list(cases)

    rng = random.Random(sample_seed)
    rng.shuffle(selected)

    if sample_size <= 0:
        return selected

    return selected[: min(sample_size, len(selected))]


def sample_fingerprint(
    cases: list[I2PCase],
) -> str:
    payload = ",".join(
        str(case.index)
        for case in cases
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()[:16]


def parse_strengths(
    config: DictConfig,
) -> list[float]:
    """
    TABLE1_STRENGTHS examples:

        500
        250,500
    """

    raw = os.environ.get(
        "TABLE1_STRENGTHS"
    )

    if raw is None:
        return [
            float(value)
            for value in config.experiment.strengths
        ]

    strengths = [
        float(part.strip())
        for part in raw.split(",")
        if part.strip()
    ]

    if not strengths:
        raise ValueError(
            "TABLE1_STRENGTHS did not contain "
            "any strengths."
        )

    return strengths


def apply_strength_override(
    config: DictConfig,
    strengths: list[float],
) -> None:
    with open_dict(config):
        config.experiment.strengths = (
            OmegaConf.create(strengths)
        )

        for schedule in config.experiment.schedules:
            if str(schedule.name) == "full_shift":
                schedule.strengths = (
                    OmegaConf.create(strengths)
                )


def parse_boolean_environment(
    name: str,
    default: bool = False,
) -> bool:
    raw = os.environ.get(name)

    if raw is None:
        return default

    normalized = raw.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be one of true/false, yes/no, on/off, or 1/0; "
        f"received {raw!r}."
    )


def apply_baseline_only_override(
    config: DictConfig,
    baseline_only: bool,
) -> None:
    if not baseline_only:
        return

    with open_dict(config):
        config.experiment.schedules = OmegaConf.create([])


@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="table1_i2p_quick",
)
def main(config: DictConfig) -> None:
    benchmark = OmegaConf.to_container(
        config.benchmark,
        resolve=True,
    )

    if not isinstance(benchmark, dict):
        raise TypeError(
            "benchmark config must resolve "
            "to a mapping."
        )

    data_path = Path(
        str(benchmark["data_path"])
    )

    sample_size = int(
        benchmark["sample_size"]
    )

    sample_seed = int(
        benchmark["sample_seed"]
    )

    worker_id = int(
        benchmark["worker_id"]
    )

    num_workers = int(
        benchmark["num_workers"]
    )

    output_root = Path(
        str(benchmark["output_root"])
    )

    if num_workers <= 0:
        raise ValueError(
            "benchmark.num_workers must be positive"
        )

    if not 0 <= worker_id < num_workers:
        raise ValueError(
            f"worker_id={worker_id} is invalid for "
            f"num_workers={num_workers}"
        )

    strengths = parse_strengths(config)

    baseline_only = parse_boolean_environment(
        "TABLE1_BASELINE_ONLY"
    )

    apply_strength_override(
        config=config,
        strengths=strengths,
    )

    apply_baseline_only_override(
        config=config,
        baseline_only=baseline_only,
    )

    all_cases = load_i2p_csv(data_path)

    selected_cases = select_cases(
        cases=all_cases,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )

    worker_cases = selected_cases[
        worker_id::num_workers
    ]

    fingerprint = sample_fingerprint(
        selected_cases
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "I2P dataset size: %d",
        len(all_cases),
    )

    logger.info(
        "Selected sample size: %d",
        len(selected_cases),
    )

    logger.info(
        "Sample seed: %d",
        sample_seed,
    )

    logger.info(
        "Sample fingerprint: %s",
        fingerprint,
    )

    logger.info(
        "Worker %d/%d receives %d cases",
        worker_id,
        num_workers,
        len(worker_cases),
    )

    logger.info(
        "Strengths: %s",
        strengths,
    )

    logger.info(
        "Baseline-only screening: %s",
        baseline_only,
    )

    logger.info(
        "Resolved configuration:\n%s",
        OmegaConf.to_yaml(
            config,
            resolve=True,
        ),
    )

    device = torch.device(
        str(config.device)
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    logger.info(
        "CUDA device: %s",
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "none"
        ),
    )

    # Deterministic state before model construction.
    set_random_seed(sample_seed)

    run_name = (
        "table1_i2p"
        f"__worker_{worker_id}"
        f"__sample_{len(selected_cases)}"
        f"__{fingerprint}"
    )

    with RunManifest(
        output_dir=str(config.run_dir),
        run_name=run_name,
        config=config,
        device=device,
    ) as manifest:
        manifest.add_result(
            "benchmark",
            {
                "dataset_size": len(all_cases),
                "selected_size": len(selected_cases),
                "worker_size": len(worker_cases),
                "sample_seed": sample_seed,
                "sample_fingerprint": fingerprint,
                "worker_id": worker_id,
                "num_workers": num_workers,
                "strengths": strengths,
                "baseline_only": baseline_only,
            },
        )

        # -------------------------------------------------
        # Create SHIFT manager once for this GPU worker.
        # -------------------------------------------------

        intervention_manager = instantiate(
            config.intervention
        )

        # -------------------------------------------------
        # Load FLUX only once for this GPU worker.
        # -------------------------------------------------

        model = instantiate(
            config.model,
            device=device,
            intervention_manager=(
                intervention_manager
            ),
            _recursive_=False,
        )

        with manifest.stage(
            "model_prepare"
        ):
            model.prepare_for_inference()

            manifest.add_result(
                "model",
                model.get_model_report(),
            )

        completed_cases = 0

        # -------------------------------------------------
        # Each I2P case receives ITS OWN sd_seed.
        #
        # The existing SteeringExperimentPipeline uses
        # self.seed to create torch.Generator. Therefore
        # we instantiate one lightweight pipeline object
        # per prompt while reusing the already loaded model.
        # -------------------------------------------------

        with manifest.stage(
            "i2p_generation"
        ):
            for local_index, case in enumerate(
                worker_cases
            ):
                logger.info(
                    "[worker %d] case %d/%d: "
                    "%s, seed=%d",
                    worker_id,
                    local_index + 1,
                    len(worker_cases),
                    case.name,
                    case.seed,
                )

                set_random_seed(case.seed)

                cases = [
                    {
                        "name": case.name,
                        "prompt": case.prompt,
                        "operation": "erase",
                    }
                ]

                task_dir = (
                    output_root
                    / case.name
                )

                benchmark_dir = (
                    task_dir
                    / "benchmark"
                )

                pipeline = instantiate(
                    config.pipeline,
                    model=model,
                    intervention_manager=(
                        intervention_manager
                    ),
                    cases=OmegaConf.create(
                        cases
                    ),
                    schedules=(
                        config.experiment.schedules
                    ),
                    strengths=(
                        config.experiment.strengths
                    ),
                    generation_config=(
                        config.generation
                    ),
                    output_dir=str(
                        benchmark_dir
                    ),
                    resume_config=(
                        config.experiment.resume
                    ),
                    seed=case.seed,
                    logger=logger,
                    _recursive_=False,
                )

                pipeline.run()

                completed_cases += 1

        manifest.add_result(
            "completed_cases",
            completed_cases,
        )

        manifest.add_result(
            "intervention_report",
            model.get_intervention_report(),
        )

    logger.info(
        "Worker %d finished %d I2P cases.",
        worker_id,
        completed_cases,
    )


if __name__ == "__main__":
    main()
