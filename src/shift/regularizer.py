from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf

from src.utils.hashing import sha256_file_set

ClassifierLocation = tuple[int, int]
EXPECTED_ACTIVATION_LOCATION = "transformer_block_output_text"


class BlockwiseSVMRegularizer:
    """
    Applies a block/step-specific SVM classifier to the
    current DiT text-token activation.

    shared:
        Runtime location (block, step) uses the classifier
        from (block, source_step).

    per_step:
        Runtime location (block, step) uses the classifier
        from the same (block, step).

    At inference time:

        activation [1, tokens, channels]
            -> mean over tokens
        feature [1, channels]
            -> predict_proba
        p_cls
            -> eta_cls
    """

    VALID_TIMING_MODES = {
        "shared",
        "per_step",
    }

    def __init__(
        self,
        classifier_directory: str,
        block_indices: Sequence[int],
        step_indices: Sequence[int] | None = None,
        timing_mode: str = "per_step",
        source_step: int = 0,
        eps: float = 1.0e-6,
        eta_max: float = 20.0,
        expected_activation_location: str = EXPECTED_ACTIVATION_LOCATION,
    ) -> None:
        self.classifier_directory = Path(classifier_directory)

        self.block_indices = sorted({int(value) for value in block_indices})

        self.step_indices = (
            sorted({int(value) for value in step_indices}) if step_indices is not None else []
        )

        self.timing_mode = str(timing_mode).strip()

        self.source_step = int(source_step)
        self.eps = float(eps)
        self.eta_max = float(eta_max)
        self.expected_activation_location = str(expected_activation_location)

        if not self.classifier_directory.is_dir():
            raise FileNotFoundError(
                "Classifier directory does not exist: " f"{self.classifier_directory}"
            )

        if not self.block_indices:
            raise ValueError("At least one classifier block is required.")

        if self.timing_mode not in self.VALID_TIMING_MODES:
            raise ValueError(
                f"Unsupported classifier timing_mode="
                f"{self.timing_mode!r}. Available: "
                f"{sorted(self.VALID_TIMING_MODES)}"
            )

        if self.timing_mode == "per_step" and not self.step_indices:
            raise ValueError("step_indices is required when classifier " "timing_mode='per_step'.")

        if self.eps <= 0:
            raise ValueError("eps must be positive.")

        if self.eta_max <= 0:
            raise ValueError("eta_max must be positive.")

        self._metadata = self._validate_metadata()

        source_steps = [self.source_step] if self.timing_mode == "shared" else self.step_indices

        self._classifiers: dict[
            ClassifierLocation,
            Any,
        ] = {}

        self._classifier_paths: dict[
            ClassifierLocation,
            str,
        ] = {}

        self._positive_class_indices: dict[
            ClassifierLocation,
            int,
        ] = {}

        for block_index in self.block_indices:
            for step_index in source_steps:
                location = (
                    block_index,
                    step_index,
                )

                self._load_classifier(location)

        self.artifact_fingerprint = sha256_file_set(
            (
                f"block={block},step={step}",
                path,
            )
            for (
                block,
                step,
            ), path in self._classifier_paths.items()
        )

    def _validate_metadata(self) -> dict[str, Any]:
        metadata_path = self.classifier_directory / "metadata.yaml"

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Classifier metadata is missing: {metadata_path}. "
                "Retrain the block-output SVM artifacts before inference."
            )

        document = OmegaConf.to_container(
            OmegaConf.load(metadata_path),
            resolve=True,
        )

        if not isinstance(document, dict):
            raise RuntimeError(f"Invalid classifier metadata: {metadata_path}")

        location = str(document.get("activation_location", "missing"))

        if location != self.expected_activation_location:
            raise RuntimeError(
                "Incompatible classifier activation location: "
                f"expected {self.expected_activation_location!r}, got {location!r}. "
                "Do not use classifiers trained from attn.to_add_out features."
            )

        normalization = str(document.get("feature_normalization", "missing"))

        if normalization != "sample_l2":
            raise RuntimeError(
                "Incompatible SVM feature normalization: "
                f"expected 'sample_l2', got {normalization!r}. "
                "Retrain with trainer.l2_normalize=true."
            )

        return document

    def _load_classifier(
        self,
        location: ClassifierLocation,
    ) -> None:
        block_index, step_index = location

        path = (
            self.classifier_directory
            / f"block_{block_index:02d}"
            / f"step_{step_index:02d}_classifier.joblib"
        )

        if not path.is_file():
            raise FileNotFoundError("Classifier for location " f"{location} does not exist: {path}")

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
                "Positive class label 1 is absent for " f"location {location}: {classes}"
            ) from error

        self._classifiers[location] = classifier
        self._classifier_paths[location] = str(path)

        self._positive_class_indices[location] = positive_index

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

    def resolve_source_location(
        self,
        block_index: int,
        runtime_step: int,
    ) -> ClassifierLocation:
        block_index = int(block_index)
        runtime_step = int(runtime_step)

        source_step = self.source_step if self.timing_mode == "shared" else runtime_step

        location = (
            block_index,
            source_step,
        )

        if location not in self._classifiers:
            raise KeyError(
                "No classifier resolves runtime location "
                f"(block={block_index}, "
                f"step={runtime_step})."
            )

        return location

    def validate_locations(
        self,
        blocks: Sequence[int] | None,
        steps: Sequence[int] | None,
    ) -> None:
        blocks = (
            sorted(self.available_blocks) if blocks is None else [int(value) for value in blocks]
        )

        if steps is None:
            if self.timing_mode == "shared":
                return

            raise ValueError(
                "Explicit runtime steps are required for " "per-step classifier timing."
            )

        missing: list[tuple[int, int]] = []

        for block_index in blocks:
            for step_index in steps:
                try:
                    self.resolve_source_location(
                        block_index=block_index,
                        runtime_step=int(step_index),
                    )
                except KeyError:
                    missing.append(
                        (
                            block_index,
                            int(step_index),
                        )
                    )

        if missing:
            raise ValueError("No classifiers resolve runtime locations: " f"{missing}")

    def predict(
        self,
        block_index: int,
        step_index: int,
        activation: torch.Tensor,
    ) -> tuple[
        float,
        float,
        ClassifierLocation,
    ]:
        """
        Return:
            p_cls:
                Probability of class 1.

            eta_cls:
                Clipped SHIFT multiplier.

            source_location:
                Classifier block/step used.
        """
        source_location = self.resolve_source_location(
            block_index=block_index,
            runtime_step=step_index,
        )

        if activation.ndim != 3:
            raise RuntimeError(
                "Expected activation shape "
                "[batch, tokens, channels], got "
                f"{tuple(activation.shape)}."
            )

        if activation.shape[0] != 1:
            raise RuntimeError(
                "Dynamic SVM steering currently expects "
                f"batch_size=1, got "
                f"{activation.shape[0]}."
            )

        # Pool on GPU first. Only one channel vector is
        # transferred to CPU.
        pooled = activation.detach().float().mean(dim=1).cpu().numpy()

        if not np.isfinite(pooled).all():
            nan_count = int(np.isnan(pooled).sum())
            inf_count = int(np.isinf(pooled).sum())
            raise FloatingPointError(
                "Non-finite FLUX activation reached the SVM at "
                f"block={block_index}, runtime_step={step_index}, "
                f"dtype={activation.dtype}, nan_values={nan_count}, "
                f"inf_values={inf_count}. This usually means the preceding "
                "steering pass overflowed FP16; use SHIFT_MODEL_DTYPE=float32 "
                "or lower TABLE1_ETA_MAX."
            )

        classifier = self._classifiers[source_location]

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

        positive_index = self._positive_class_indices[source_location]

        if hasattr(classifier, "predict_member_probabilities"):
            probabilities = classifier.predict_member_probabilities(pooled)
            positive_probabilities = probabilities[:, 0, positive_index]
        else:
            probabilities = classifier.predict_proba(pooled)
            positive_probabilities = probabilities[:, positive_index]

        positive_probabilities = np.clip(positive_probabilities, 0.0, 1.0)
        p_cls = float(np.mean(positive_probabilities))

        # The released implementation converts each ensemble member's
        # probability to odds first, clips it, and only then averages.
        member_eta = 1.0 / ((1.0 - positive_probabilities) + self.eps) - 1.0
        member_eta = np.clip(member_eta, 0.0, self.eta_max)
        eta_cls = float(np.mean(member_eta))

        return (
            p_cls,
            eta_cls,
            source_location,
        )

    @property
    def available_blocks(
        self,
    ) -> set[int]:
        return {block_index for block_index, _ in self._classifiers}

    def path_for(
        self,
        location: ClassifierLocation,
    ) -> str:
        return self._classifier_paths[location]

    def configuration(self) -> dict[str, Any]:
        return {
            "classifier_directory": str(self.classifier_directory),
            "timing_mode": self.timing_mode,
            "source_step": (self.source_step if self.timing_mode == "shared" else None),
            "step_indices": (list(self.step_indices) if self.timing_mode == "per_step" else None),
            "eps": self.eps,
            "eta_max": self.eta_max,
            "activation_location": self._metadata["activation_location"],
            "feature_normalization": self._metadata["feature_normalization"],
            "ensemble_size": self._metadata.get("ensemble_size"),
            "artifact_fingerprint": (
                self.artifact_fingerprint
            ),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            **self.configuration(),
            "available_blocks": sorted(self.available_blocks),
            "classifiers": [
                {
                    "block": block_index,
                    "step": step_index,
                    "path": self._classifier_paths[(block_index, step_index)],
                    "positive_class_index": (
                        self._positive_class_indices[(block_index, step_index)]
                    ),
                }
                for block_index, step_index in sorted(self._classifiers)
            ],
        }
