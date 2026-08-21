from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import hydra
import torch

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.benchmarks.snoopy import build_cases, resolve_task
from src.utils.init_utils import set_random_seed
from src.utils.run_manifest import RunManifest

logger = logging.getLogger(__name__)


def get_worker_task_ids(
    worker_id: int,
    num_workers: int,
    total_tasks: int,
) -> list[int]:
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    if worker_id < 0:
        raise ValueError("worker_id must be >= 0")

    if worker_id >= num_workers:
        raise ValueError(f"worker_id={worker_id} must be " f"< num_workers={num_workers}")

    return list(range(worker_id, total_tasks, num_workers))


@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="snoopy_benchmark",
)
def main(config: DictConfig) -> None:
    benchmark_config = OmegaConf.to_container(
        config.benchmark,
        resolve=True,
    )

    concepts = [dict(item) for item in benchmark_config["concepts"]]
    seeds = [int(seed) for seed in benchmark_config["seeds"]]
    total_tasks = len(concepts) * len(seeds)
    worker_id = int(benchmark_config["worker_id"])
    num_workers = int(benchmark_config["num_workers"])

    task_ids = get_worker_task_ids(
        worker_id=worker_id,
        num_workers=num_workers,
        total_tasks=total_tasks,
    )

    logger.info("Worker %d/%d", worker_id, num_workers)
    logger.info("Logical task IDs: %s", task_ids)
    logger.info("Logical tasks: %d", len(task_ids))

    device = torch.device(str(config.device))

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    logger.info(
        "GPU: %s",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    )

    # Load intervention manager and FLUX exactly ONCE.
    logger.info("Creating intervention manager")

    intervention_manager = instantiate(config.intervention)

    logger.info("Loading FLUX")

    model = instantiate(
        config.model,
        device=device,
        intervention_manager=(intervention_manager),
        _recursive_=False,
    )

    logger.info("Preparing FLUX for inference")

    model.prepare_for_inference()

    logger.info("FLUX ready")

    # Process logical experiment tasks
    output_root = Path(str(benchmark_config["output_root"]))
    num_templates = int(benchmark_config["num_templates"])
    completed_tasks = []

    for task_position, task_id in enumerate(task_ids, start=1):
        task = resolve_task(task_id=task_id, concepts=concepts, seeds=seeds)

        logger.info(
            "Worker %d: logical task " "%d/%d",
            worker_id,
            task_position,
            len(task_ids),
        )

        logger.info(
            "task_id=%d concept=%s seed=%d",
            task.task_id,
            task.concept_key,
            task.seed,
        )

        set_random_seed(task.seed)

        cases = build_cases(
            concept_key=(task.concept_key),
            concept_text=(task.concept_text),
            num_templates=(num_templates),
        )

        # Task output directory
        # Keep task_N layout as evaluation script wants
        task_dir = output_root / f"task_{task.task_id}"
        experiment_dir = task_dir / "benchmark"
        task_dir.mkdir(parents=True, exist_ok=True)

        # Manifest for this logical benchmark task
        run_name = "snoopy_table3" f"__{task.concept_key}" f"__seed_{task.seed}"

        # Make a task-specific copy of the Hydra config.
        task_config = OmegaConf.create(
            OmegaConf.to_container(
                config,
                resolve=True,
            )
        )

        task_config.seed = task.seed
        task_config.experiment.cases = OmegaConf.create(cases)

        with RunManifest(
            output_dir=str(task_dir),
            run_name=run_name,
            config=task_config,
            device=device,
        ) as manifest:
            manifest.add_result("benchmark_task", asdict(task))

            pipeline = instantiate(
                config.pipeline,
                model=model,
                intervention_manager=(intervention_manager),
                cases=cases,
                schedules=(config.experiment.schedules),
                strengths=(config.experiment.strengths),
                generation_config=(config.generation),
                output_dir=str(experiment_dir),
                resume_config=(config.experiment.resume),
                seed=task.seed,
                logger=logger,
                _recursive_=False,
            )

            results = pipeline.run()

            manifest.add_result("experiment", results)
            completed_tasks.append(task.task_id)

    logger.info("worker_id=%d finished", worker_id)
    logger.info("completed logical tasks=%s", completed_tasks)


if __name__ == "__main__":
    main()
