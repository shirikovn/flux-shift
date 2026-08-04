from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch


class BlockwiseSVMRegularizer:
    """
    Loads one block-specific sklearn classifier per FLUX block.

    At inference time:

        activation [1, tokens, channels]
            ↓ mean over tokens
        pooled [1, channels]
            ↓ SVM predict_proba
        p_cls
            ↓
        eta = clip(1 / (1 - p_cls + eps) - 1, 0, eta_max)
    """

    def __init__(
        self,
        classifier_directory: str,
        block_indices: Sequence[int],
        source_step: int = 0,
        eps: float = 1.0e-6,
        eta_max: float = 4.0,
    ) -> None:
        self.classifier_directory = Path(classifier_directory)

        self.block_indices = [int(value) for value in block_indices]

        self.source_step = int(source_step)
        self.eps = float(eps)
        self.eta_max = float(eta_max)

        if not self.classifier_directory.is_dir():
            raise FileNotFoundError(
                "Classifier directory does not exist: " f"{self.classifier_directory}"
            )

        if not self.block_indices:
            raise ValueError("At least one classifier block is required.")

        if self.eps <= 0:
            raise ValueError("eps must be positive.")

        if self.eta_max <= 0:
            raise ValueError("eta_max must be positive.")

        self._classifiers: dict[int, Any] = {}
        self._classifier_paths: dict[int, str] = {}
        self._positive_class_indices: dict[
            int,
            int,
        ] = {}

        for block_index in self.block_indices:
            path = (
                self.classifier_directory
                / f"block_{block_index:02d}"
                / (f"step_{self.source_step:02d}" "_classifier.joblib")
            )

            if not path.is_file():
                raise FileNotFoundError(
                    f"Classifier for block {block_index} " f"does not exist: {path}"
                )

            classifier = joblib.load(path)

            if not hasattr(
                classifier,
                "predict_proba",
            ):
                raise TypeError(f"Classifier in {path} has no " "predict_proba method.")

            classes = self._get_classes(classifier)

            try:
                positive_index = classes.index(1)
            except ValueError as error:
                raise RuntimeError(
                    "Positive class label 1 is absent " f"for block {block_index}: {classes}"
                ) from error

            self._classifiers[block_index] = classifier

            self._classifier_paths[block_index] = str(path)

            self._positive_class_indices[block_index] = positive_index

    @staticmethod
    def _get_classes(
        classifier: Any,
    ) -> list[Any]:
        classes = getattr(
            classifier,
            "classes_",
            None,
        )

        if classes is None:
            # Fallback for an sklearn Pipeline.
            steps = getattr(
                classifier,
                "steps",
                None,
            )

            if steps:
                final_estimator = steps[-1][1]
                classes = getattr(
                    final_estimator,
                    "classes_",
                    None,
                )

        if classes is None:
            raise RuntimeError("Could not find classifier classes_.")

        return np.asarray(classes).tolist()

    @property
    def available_blocks(self) -> set[int]:
        return set(self._classifiers)

    def predict(
        self,
        block_index: int,
        activation: torch.Tensor,
    ) -> tuple[float, float]:
        """
        Return:
            p_cls: probability of class 1;
            eta: clipped SHIFT scaling coefficient.
        """
        if block_index not in self._classifiers:
            raise KeyError(f"No classifier loaded for block " f"{block_index}.")

        if activation.ndim != 3:
            raise RuntimeError(
                "Expected activation shape "
                "[batch, tokens, channels], got "
                f"{tuple(activation.shape)}."
            )

        if activation.shape[0] != 1:
            raise RuntimeError(
                "Dynamic SVM steering currently expects "
                f"batch_size=1, got {activation.shape[0]}."
            )

        # Pool on GPU first. Only one 3072-dimensional
        # vector is then copied to CPU.
        pooled = activation.detach().float().mean(dim=1).cpu().numpy()

        classifier = self._classifiers[block_index]

        expected_features = getattr(
            classifier,
            "n_features_in_",
            None,
        )

        if expected_features is not None and pooled.shape[1] != int(expected_features):
            raise RuntimeError(
                "Classifier feature dimension mismatch: "
                f"activation={pooled.shape[1]}, "
                f"classifier={expected_features}."
            )

        probabilities = classifier.predict_proba(pooled)

        positive_index = self._positive_class_indices[block_index]

        p_cls = float(probabilities[0, positive_index])

        # Protect against tiny numerical excursions
        # outside [0, 1].
        p_cls = float(np.clip(p_cls, 0.0, 1.0))

        eta = 1.0 / ((1.0 - p_cls) + self.eps) - 1.0

        eta = float(
            np.clip(
                eta,
                0.0,
                self.eta_max,
            )
        )

        return p_cls, eta

    def summary(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "classifier_directory": str(self.classifier_directory),
            "source_step": self.source_step,
            "eps": self.eps,
            "eta_max": self.eta_max,
            "available_blocks": sorted(self.available_blocks),
            "classifiers": {
                str(block_index): {
                    "path": (self._classifier_paths[block_index]),
                    "positive_class_index": (self._positive_class_indices[block_index]),
                }
                for block_index in sorted(self._classifiers)
            },
        }
