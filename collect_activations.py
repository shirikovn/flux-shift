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
    config_name="collect_activations",
)
def main(config: DictConfig) -> None:
    logger.info(
        "Resolved configuration:\n%s",
        OmegaConf.to_yaml(config, resolve=True),
    )

    set_random_seed(int(config.seed))

    device = torch.device(str(config.device))

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "config.device=cuda, but CUDA is unavailable."
        )

    if device.type == "cuda":
        logger.info("CUDA runtime: %s", torch.version.cuda)
        logger.info("GPU: %s", torch.cuda.get_device_name(device))

    intervention_manager = instantiate(config.intervention)
    dataset = instantiate(config.dataset)

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
        intervention_manager=intervention_manager,
        dataset=dataset,
        generation_config=config.generation,
        output_dir=str(config.output_dir),
        seed=int(config.seed),
        logger=logger,
        _recursive_=False,
    )

    results = pipeline.collect()

    logger.info(
        "Collection results:\n%s",
        OmegaConf.to_yaml(OmegaConf.create(results), resolve=True),
    )

    report = model.get_intervention_report()
    logger.info(
        "Intervention report:\n%s",
        OmegaConf.to_yaml(OmegaConf.create(report), resolve=True),
    )


if __name__ == "__main__":
    main()
