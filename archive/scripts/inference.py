from __future__ import annotations

import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.utils.init_utils import set_random_seed

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="flux_inference",
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

    if device.type == "cuda":
        logger.info(
            "CUDA runtime: %s",
            torch.version.cuda,
        )
        logger.info(
            "GPU: %s",
            torch.cuda.get_device_name(device),
        )

    intervention_manager = instantiate(
        config.intervention,
    )

    model = instantiate(
        config.model,
        device=device,
        intervention_manager=intervention_manager,
        _recursive_=False,
    )
    model.prepare_for_inference()

    pipeline = instantiate(
        config.pipeline,
        model=model,
        prompt=str(config.prompt),
        seed=int(config.seed),
        generation_config=config.generation,
        output_config=config.output,
        logger=logger,
        _recursive_=False,
    )

    output_paths = pipeline.generate()

    report = model.get_intervention_report()

    logger.info(
        "Intervention report:\n%s",
        OmegaConf.to_yaml(
            OmegaConf.create(report),
            resolve=True,
        ),
    )

    for output_path in output_paths:
        logger.info(
            "Saved image: %s",
            output_path,
        )


if __name__ == "__main__":
    main()
