from __future__ import annotations

import logging

import hydra
from hydra.utils import instantiate
from omegaconf import (
    DictConfig,
    OmegaConf,
)


logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="train_svm",
)
def main(config: DictConfig) -> None:
    logger.info(
        "Resolved configuration:\n%s",
        OmegaConf.to_yaml(
            config,
            resolve=True,
        ),
    )

    trainer = instantiate(
        config.trainer
    )

    results = trainer.run()

    logger.info(
        "SVM training results:\n%s",
        OmegaConf.to_yaml(
            OmegaConf.create(results),
            resolve=True,
        ),
    )


if __name__ == "__main__":
    main()
