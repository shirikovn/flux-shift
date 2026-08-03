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

    random_seed = int(config.trainer.random_seed)
    set_random_seed(random_seed)

    # SVM training is CPU-based. The manifest will therefore
    # record CPU runtime and no CUDA memory statistics.
    device = torch.device("cpu")

    with RunManifest(
        output_dir=str(config.run_dir),
        run_name="train_svm",
        config=config,
        device=device,
    ) as manifest:
        with manifest.stage("svm_training"):
            trainer = instantiate(config.trainer)
            results = trainer.run()

        manifest.add_result(
            "svm_training",
            results,
        )

        logger.info(
            "SVM training results:\n%s",
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
