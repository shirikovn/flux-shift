from __future__ import annotations

import logging
from dataclasses import asdict

import hydra
import torch

from hydra.utils import instantiate

from omegaconf import (
    DictConfig,
    OmegaConf,
    open_dict,
)

from src.benchmarks.snoopy import (
    build_cases,
    resolve_task,
)

from src.utils.init_utils import (
    set_random_seed,
)

from src.utils.run_manifest import (
    RunManifest,
)


logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="snoopy_benchmark",
)
def main(config: DictConfig) -> None:

    # -----------------------------------------------------
    # Resolve benchmark task
    # -----------------------------------------------------

    benchmark_config = OmegaConf.to_container(
        config.benchmark,
        resolve=True,
    )

    if not isinstance(
        benchmark_config,
        dict,
    ):
        raise TypeError(
            "benchmark config must resolve "
            "to a mapping."
        )


    raw_concepts = benchmark_config[
        "concepts"
    ]

    raw_seeds = benchmark_config[
        "seeds"
    ]


    if not isinstance(
        raw_concepts,
        list,
    ):
        raise TypeError(
            "benchmark.concepts must be a list."
        )

    if not isinstance(
        raw_seeds,
        list,
    ):
        raise TypeError(
            "benchmark.seeds must be a list."
        )


    concepts = [
        dict(concept)
        for concept in raw_concepts
    ]

    seeds = [
        int(seed)
        for seed in raw_seeds
    ]


    task = resolve_task(
        task_id=int(
            benchmark_config["task_id"]
        ),
        concepts=concepts,
        seeds=seeds,
    )


    cases = build_cases(
        concept_key=task.concept_key,
        concept_text=task.concept_text,
        num_templates=int(
            benchmark_config[
                "num_templates"
            ]
        ),
    )


    # -----------------------------------------------------
    # Put task-specific data into resolved Hydra config.
    # -----------------------------------------------------

    with open_dict(config):

        config.seed = task.seed

        config.experiment.cases = (
            OmegaConf.create(cases)
        )


    logger.info(
        "Benchmark task:\n%s",
        OmegaConf.to_yaml(
            OmegaConf.create(
                asdict(task)
            ),
            resolve=True,
        ),
    )


    logger.info(
        "Number of templates: %d",
        len(cases),
    )


    logger.info(
        "Resolved configuration:\n%s",
        OmegaConf.to_yaml(
            config,
            resolve=True,
        ),
    )


    # -----------------------------------------------------
    # Random state / device
    # -----------------------------------------------------

    set_random_seed(task.seed)

    device = torch.device(
        str(config.device)
    )


    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "config.device=cuda, "
            "but CUDA is unavailable."
        )


    logger.info(
        "CUDA device: %s",
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "none",
    )


    # -----------------------------------------------------
    # Manifest
    # -----------------------------------------------------

    run_name = (
        "snoopy_table3"
        f"__{task.concept_key}"
        f"__seed_{task.seed}"
    )


    with RunManifest(
        output_dir=str(config.run_dir),
        run_name=run_name,
        config=config,
        device=device,
    ) as manifest:

        manifest.add_result(
            "benchmark_task",
            asdict(task),
        )


        # -------------------------------------------------
        # Steering manager
        # -------------------------------------------------

        intervention_manager = (
            instantiate(
                config.intervention
            )
        )


        # -------------------------------------------------
        # Model
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


        # -------------------------------------------------
        # Benchmark pipeline
        # -------------------------------------------------

        pipeline = instantiate(
            config.pipeline,

            model=model,

            intervention_manager=(
                intervention_manager
            ),

            cases=config.experiment.cases,

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
                config.experiment.output_dir
            ),

            resume_config=(
                config.experiment.resume
            ),

            seed=task.seed,

            logger=logger,

            _recursive_=False,
        )


        with manifest.stage(
            "snoopy_table3"
        ):
            results = pipeline.run()


        manifest.add_result(
            "experiment",
            results,
        )


        manifest.add_result(
            "intervention_report",
            model.get_intervention_report(),
        )


        logger.info(
            "Benchmark completed:\n%s",
            OmegaConf.to_yaml(
                OmegaConf.create(
                    results
                ),
                resolve=True,
            ),
        )


        logger.info(
            "Run manifest: %s",
            manifest.path,
        )


if __name__ == "__main__":
    main()
