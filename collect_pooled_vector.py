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
    config_name="collect_pooled_vector",
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

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config.device=cuda, but CUDA is unavailable.")

    with RunManifest(
        output_dir=str(config.run_dir),
        run_name="collect_pooled_vector",
        config=config,
        device=device,
    ) as manifest:
        dataset = instantiate(config.dataset)

        model = instantiate(
            config.model,
            device=device,
            intervention_manager=None,
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
            dataset=dataset,
            target_prompt=str(config.target_prompt),
            generation_config=config.generation,
            output_dir=str(config.output_dir),
            logger=logger,
            _recursive_=False,
        )

        with manifest.stage("pooled_vector_collection"):
            results = pipeline.collect()

        manifest.add_result(
            "pooled_collection",
            results,
        )

        logger.info(
            "Pooled collection results:\n%s",
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
