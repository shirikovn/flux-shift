from __future__ import annotations

import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.utils.init_utils import set_random_seed
from src.utils.run_manifest import RunManifest


logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="full_shift_experiment",
)
def main(config: DictConfig) -> None:
    logger.info(
        "Resolved configuration:\n%s",
        OmegaConf.to_yaml(
            config,
            resolve=True,
        ),
    )

    set_random_seed(int(config.seed))

    device = torch.device(str(config.device))

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "config.device=cuda, but CUDA is unavailable."
        )

    with RunManifest(
        output_dir=str(config.run_dir),
        run_name="full_shift_experiment",
        config=config,
        device=device,
    ) as manifest:
        intervention_manager = instantiate(
            config.intervention
        )

        model = instantiate(
            config.model,
            device=device,
            intervention_manager=intervention_manager,
            _recursive_=False,
        )

        with manifest.stage("model_prepare"):
            model.prepare_for_inference()

        manifest.add_result(
            "model",
            model.get_model_report(),
        )

        pipeline = instantiate(
            config.pipeline,
            model=model,
            intervention_manager=intervention_manager,
            cases=config.experiment.cases,
            schedules=config.experiment.schedules,
            strengths=config.experiment.strengths,
            generation_config=config.generation,
            output_dir=str(
                config.experiment.output_dir
            ),
            resume_config=config.experiment.resume,
            seed=int(config.seed),
            logger=logger,
            _recursive_=False,
        )

        with manifest.stage("steering_experiment"):
            results = pipeline.run()

        intervention_report = (
            model.get_intervention_report()
        )

        manifest.add_result(
            "experiment",
            results,
        )
        manifest.add_result(
            "intervention_report",
            intervention_report,
        )

        logger.info(
            "Experiment results:\n%s",
            OmegaConf.to_yaml(
                OmegaConf.create(results),
                resolve=True,
            ),
        )
        logger.info(
            "Run manifest: %s",
            manifest.path,
        )


if __name__ == "__main__":
    main()
