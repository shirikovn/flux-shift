from __future__ import annotations

import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


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

    device = torch.device(
        str(config.device)
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is unavailable."
        )

    dataset = instantiate(
        config.dataset
    )

    model = instantiate(
        config.model,
        device=device,
        intervention_manager=None,
        _recursive_=False,
    )

    model.prepare_for_inference()

    pipeline = instantiate(
        config.pipeline,
        model=model,
        dataset=dataset,
        target_prompt=str(
            config.target_prompt
        ),
        generation_config=(
            config.generation
        ),
        output_dir=str(
            config.output_dir
        ),
        logger=logger,
        _recursive_=False,
    )

    results = pipeline.collect()

    logger.info(
        "Pooled collection results:\n%s",
        OmegaConf.to_yaml(
            OmegaConf.create(results),
            resolve=True,
        ),
    )


if __name__ == "__main__":
    main()
