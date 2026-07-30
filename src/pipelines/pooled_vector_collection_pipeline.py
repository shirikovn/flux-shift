from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from src.models.flux_model import FluxModel


class PooledVectorCollectionPipeline:
    """
    Build a mean pooled-CLIP difference vector from
    contrastive prompt pairs.

    direction = mean(
        positive_pooled - negative_pooled
    )

    According to the paper, the pooled direction is
    kept as a raw activation difference and is not
    channel-normalized like the DiT token vectors.
    """

    def __init__(
        self,
        model: FluxModel,
        dataset: Any,
        target_prompt: str,
        generation_config: DictConfig,
        output_dir: str,
        logger: logging.Logger,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.target_prompt = str(
            target_prompt
        )
        self.generation_config = (
            generation_config
        )
        self.output_dir = Path(
            output_dir
        )
        self.logger = logger

    def collect(self) -> dict[str, Any]:
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        difference_sum: (
            torch.Tensor | None
        ) = None

        positive_sum: (
            torch.Tensor | None
        ) = None

        negative_sum: (
            torch.Tensor | None
        ) = None

        pair_records: list[
            dict[str, Any]
        ] = []

        num_pairs = 0

        for pair in self.dataset:
            self.logger.info(
                "Encoding pooled pair: %s",
                pair.name,
            )

            negative = (
                self.model
                .encode_pooled_prompt(
                    prompt=(
                        pair.negative_prompt
                    ),
                    max_sequence_length=int(
                        self.generation_config
                        .max_sequence_length
                    ),
                )
                .detach()
                .float()
                .cpu()
            )

            positive = (
                self.model
                .encode_pooled_prompt(
                    prompt=(
                        pair.positive_prompt
                    ),
                    max_sequence_length=int(
                        self.generation_config
                        .max_sequence_length
                    ),
                )
                .detach()
                .float()
                .cpu()
            )

            if negative.shape != positive.shape:
                raise RuntimeError(
                    "Positive and negative pooled "
                    "embedding shapes differ: "
                    f"{positive.shape} != "
                    f"{negative.shape}."
                )

            if negative.ndim != 2:
                raise RuntimeError(
                    "Expected pooled embeddings "
                    "[batch, channels], got "
                    f"{tuple(negative.shape)}."
                )

            if negative.shape[0] != 1:
                raise RuntimeError(
                    "Pooled collection currently "
                    "expects batch_size=1."
                )

            difference = (
                positive - negative
            )

            if not torch.isfinite(
                difference
            ).all():
                raise RuntimeError(
                    "Pooled difference contains "
                    "NaN or Inf."
                )

            if difference_sum is None:
                difference_sum = (
                    torch.zeros_like(
                        difference
                    )
                )

                positive_sum = (
                    torch.zeros_like(
                        positive
                    )
                )

                negative_sum = (
                    torch.zeros_like(
                        negative
                    )
                )

            difference_sum.add_(
                difference
            )

            positive_sum.add_(
                positive
            )

            negative_sum.add_(
                negative
            )

            num_pairs += 1

            pair_records.append(
                {
                    "pair_name": pair.name,
                    "negative_prompt": (
                        pair.negative_prompt
                    ),
                    "positive_prompt": (
                        pair.positive_prompt
                    ),
                    "difference_norm": float(
                        difference
                        .norm()
                        .item()
                    ),
                    "difference_mean_abs": (
                        float(
                            difference
                            .abs()
                            .mean()
                            .item()
                        )
                    ),
                }
            )

        if (
            difference_sum is None
            or positive_sum is None
            or negative_sum is None
            or num_pairs == 0
        ):
            raise RuntimeError(
                "No pooled prompt pairs were "
                "collected."
            )

        pooled_vector = (
            difference_sum
            / num_pairs
        ).squeeze(0)

        positive_mean = (
            positive_sum
            / num_pairs
        ).squeeze(0)

        negative_mean = (
            negative_sum
            / num_pairs
        ).squeeze(0)

        target_embedding = (
            self.model
            .encode_pooled_prompt(
                prompt=self.target_prompt,
                max_sequence_length=int(
                    self.generation_config
                    .max_sequence_length
                ),
            )
            .detach()
            .float()
            .cpu()
            .squeeze(0)
        )

        if (
            pooled_vector.shape
            != target_embedding.shape
        ):
            raise RuntimeError(
                "Pooled vector and target "
                "embedding shapes differ: "
                f"{pooled_vector.shape} != "
                f"{target_embedding.shape}."
            )

        vector_path = (
            self.output_dir
            / "pooled_vector.pt"
        )

        target_path = (
            self.output_dir
            / "target_embedding.pt"
        )

        positive_mean_path = (
            self.output_dir
            / "positive_mean.pt"
        )

        negative_mean_path = (
            self.output_dir
            / "negative_mean.pt"
        )

        torch.save(
            pooled_vector,
            vector_path,
        )

        torch.save(
            target_embedding,
            target_path,
        )

        torch.save(
            positive_mean,
            positive_mean_path,
        )

        torch.save(
            negative_mean,
            negative_mean_path,
        )

        cosine_vector_target = float(
            F.cosine_similarity(
                pooled_vector.unsqueeze(0),
                target_embedding.unsqueeze(0),
                dim=-1,
            )[0].item()
        )

        metadata = {
            "concept_name": (
                self.target_prompt
            ),
            "estimator": (
                "mean_raw_pooled_difference"
            ),
            "difference_direction": (
                "positive_minus_negative"
            ),
            "normalized": False,
            "num_prompt_pairs": num_pairs,
            "shape": list(
                pooled_vector.shape
            ),
            "dtype": str(
                pooled_vector.dtype
            ),
            "vector_norm": float(
                pooled_vector.norm().item()
            ),
            "vector_mean_abs": float(
                pooled_vector
                .abs()
                .mean()
                .item()
            ),
            "target_embedding_norm": float(
                target_embedding
                .norm()
                .item()
            ),
            "cosine_vector_target": (
                cosine_vector_target
            ),
            "vector_path": str(
                vector_path
            ),
            "target_embedding_path": str(
                target_path
            ),
            "positive_mean_path": str(
                positive_mean_path
            ),
            "negative_mean_path": str(
                negative_mean_path
            ),
            "pairs": pair_records,
        }

        metadata_path = (
            self.output_dir
            / "metadata.yaml"
        )

        OmegaConf.save(
            config=OmegaConf.create(
                metadata
            ),
            f=metadata_path,
        )

        self.logger.info(
            "Saved pooled vector: %s",
            vector_path,
        )

        return {
            "vector_path": str(
                vector_path
            ),
            "target_embedding_path": str(
                target_path
            ),
            "metadata_path": str(
                metadata_path
            ),
            "num_prompt_pairs": num_pairs,
        }
