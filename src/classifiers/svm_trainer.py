from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class LinearSVMTrainer:
    """
    Trains one token-pooled linear SVM classifier per FLUX block.

    Validation splitting is performed by prompt-pair name, ensuring
    that positive and negative samples from the same pair cannot be
    separated between train and validation.
    """

    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
        block_indices: Sequence[int],
        step_indices: Sequence[int] = (0,),
        validation_fraction: float = 0.2,
        random_seed: int = 123,
        c: float = 1.0,
        class_weight: str | None = "balanced",
        standardize: bool = True,
        probability: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)

        self.block_indices = sorted({int(value) for value in block_indices})

        self.step_indices = sorted({int(value) for value in step_indices})

        self.validation_fraction = float(validation_fraction)
        self.random_seed = int(random_seed)

        self.c = float(c)
        self.class_weight = class_weight
        self.standardize = bool(standardize)
        self.probability = bool(probability)

        if not self.dataset_dir.is_dir():
            raise FileNotFoundError("SVM dataset directory does not exist: " f"{self.dataset_dir}")

        if not self.block_indices:
            raise ValueError("At least one block index is required.")

        if not self.step_indices:
            raise ValueError("At least one SVM step index is required.")

        if self.step_indices[0] < 0:
            raise ValueError("SVM step indices must be nonnegative.")

        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")

        if self.c <= 0:
            raise ValueError("SVM C must be positive.")

        if not self.probability:
            raise ValueError(
                "SHIFT requires classifier probabilities, " "so probability must be enabled."
            )

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        location_results: list[dict[str, Any]] = []

        for block_index in self.block_indices:
            for step_index in self.step_indices:
                result = self._train_location(
                    block_index=block_index,
                    step_index=step_index,
                )

                location_results.append(result)

        summary = {
            "classifier_type": ("standardized_linear_svc" if self.standardize else "linear_svc"),
            "kernel": "linear",
            "probability": self.probability,
            "positive_class_label": 1,
            "negative_class_label": 0,
            "step_indices": self.step_indices,
            "validation_fraction": (self.validation_fraction),
            "random_seed": self.random_seed,
            "c": self.c,
            "class_weight": self.class_weight,
            "num_blocks": len(self.block_indices),
            "num_steps": len(self.step_indices),
            "num_locations": len(location_results),
            "locations": location_results,
        }

        summary_path = self.output_dir / "metadata.yaml"

        OmegaConf.save(
            config=OmegaConf.create(summary),
            f=summary_path,
        )

        return {
            "output_dir": str(self.output_dir),
            "metadata_path": str(summary_path),
            "num_blocks": len(self.block_indices),
            "num_steps": len(self.step_indices),
            "num_locations": len(location_results),
        }

    def _train_location(
        self,
        block_index: int,
        step_index: int,
    ) -> dict[str, Any]:
        (
            features,
            labels,
            samples,
        ) = self._load_dataset(
            block_index=block_index,
            step_index=step_index,
        )

        groups = np.asarray([str(sample["pair_name"]) for sample in samples])

        self._validate_pair_groups(
            labels=labels,
            samples=samples,
        )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.validation_fraction,
            random_state=self.random_seed,
        )

        train_indices, validation_indices = next(
            splitter.split(
                features,
                labels,
                groups=groups,
            )
        )

        train_features = features[train_indices]
        train_labels = labels[train_indices]

        validation_features = features[validation_indices]
        validation_labels = labels[validation_indices]

        evaluation_model = self._build_model()

        evaluation_model.fit(
            train_features,
            train_labels,
        )

        train_metrics = self._evaluate(
            model=evaluation_model,
            features=train_features,
            labels=train_labels,
        )

        validation_metrics = self._evaluate(
            model=evaluation_model,
            features=validation_features,
            labels=validation_labels,
        )

        train_pair_names = sorted(set(groups[train_indices].tolist()))

        validation_pair_names = sorted(set(groups[validation_indices].tolist()))

        overlap = set(train_pair_names) & set(validation_pair_names)

        if overlap:
            raise RuntimeError(
                "Pair leakage detected between " f"train and validation: {sorted(overlap)}"
            )

        # Refit the classifier on all available pairs after honest
        # holdout metrics have been computed.
        final_model = self._build_model()

        final_model.fit(
            features,
            labels,
        )

        svm_normal, svm_normal_metadata = self._extract_svm_normal(final_model)

        block_dir = self.output_dir / f"block_{block_index:02d}"

        block_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        prefix = f"step_{step_index:02d}"

        classifier_path = block_dir / f"{prefix}_classifier.joblib"
        metrics_path = block_dir / f"{prefix}_metrics.yaml"
        split_path = block_dir / f"{prefix}_split.yaml"
        svm_normal_path = block_dir / f"{prefix}_svm_normal.pt"

        joblib.dump(
            value=final_model,
            filename=classifier_path,
            compress=3,
        )

        torch.save(svm_normal, svm_normal_path)

        metrics_document = {
            "block_index": block_index,
            "step_index": step_index,
            "feature_dimension": int(features.shape[1]),
            "num_samples": int(features.shape[0]),
            "num_pairs": len(set(groups)),
            "train_num_samples": int(len(train_indices)),
            "validation_num_samples": int(len(validation_indices)),
            "evaluation_model": ("fit_on_train_split"),
            "saved_model": ("refit_on_full_dataset"),
            "train": train_metrics,
            "validation": validation_metrics,
            "classifier_path": str(classifier_path),
            "svm_normal": {
                **svm_normal_metadata,
                "path": str(svm_normal_path),
            },
        }

        OmegaConf.save(
            config=OmegaConf.create(metrics_document),
            f=metrics_path,
        )

        OmegaConf.save(
            config=OmegaConf.create(
                {
                    "train_pair_names": (train_pair_names),
                    "validation_pair_names": (validation_pair_names),
                }
            ),
            f=split_path,
        )

        return {
            "block_index": block_index,
            "step_index": step_index,
            "classifier_path": str(classifier_path),
            "svm_normal_path": str(svm_normal_path),
            "metrics_path": str(metrics_path),
            "split_path": str(split_path),
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_balanced_accuracy": (validation_metrics["balanced_accuracy"]),
            "validation_roc_auc": validation_metrics["roc_auc"],
            "validation_probability_gap": (validation_metrics["probability_gap"]),
        }

    def _build_model(self) -> Pipeline:
        pipeline_steps: list[tuple[str, Any]] = []

        if self.standardize:
            pipeline_steps.append(
                (
                    "scaler",
                    StandardScaler(),
                )
            )

        pipeline_steps.append(
            (
                "svm",
                SVC(
                    kernel="linear",
                    C=self.c,
                    class_weight=(self.class_weight),
                    probability=True,
                    random_state=(self.random_seed),
                ),
            )
        )

        return Pipeline(pipeline_steps)

    @staticmethod
    def _extract_svm_normal(
        model: Pipeline,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Extract the normalized linear-SVM normal in the original
        activation coordinate system.

        If StandardScaler is present, SVC coefficients are defined
        in standardized coordinates and must be divided by the
        per-feature scale.
        """
        svm = model.named_steps.get("svm")

        if not isinstance(svm, SVC):
            raise TypeError("Expected the pipeline step 'svm' to be an SVC.")

        classes = [int(value) for value in svm.classes_.tolist()]

        if classes != [0, 1]:
            raise RuntimeError("Expected SVM class order [0, 1], " f"received {classes}.")

        coefficients = np.asarray(
            svm.coef_,
            dtype=np.float64,
        )

        if coefficients.ndim != 2 or coefficients.shape[0] != 1:
            raise RuntimeError(
                "Expected one binary linear-SVM normal, got " f"shape {coefficients.shape}."
            )

        normal = coefficients[0].copy()
        standardized = "scaler" in model.named_steps

        if standardized:
            scaler = model.named_steps["scaler"]

            if not isinstance(scaler, StandardScaler):
                raise TypeError("Expected the pipeline step 'scaler' to be " "a StandardScaler.")

            scale = np.asarray(
                scaler.scale_,
                dtype=np.float64,
            )

            if scale.shape != normal.shape:
                raise RuntimeError(
                    "SVM normal and StandardScaler scale have "
                    "different shapes: "
                    f"normal={normal.shape}, scale={scale.shape}."
                )

            if not np.isfinite(scale).all() or np.any(scale <= 0):
                raise RuntimeError("StandardScaler contains invalid scale values.")

            normal = normal / scale

        if not np.isfinite(normal).all():
            raise RuntimeError("SVM normal contains NaN or Inf.")

        raw_norm = float(np.linalg.norm(normal))

        if raw_norm <= 1.0e-12:
            raise RuntimeError("The fitted SVM has a zero-length normal.")

        normal = normal / raw_norm

        normal_tensor = torch.from_numpy(normal.astype(np.float32))

        return normal_tensor, {
            "shape": list(normal_tensor.shape),
            "dtype": str(normal_tensor.dtype),
            "space": "raw_activation_space",
            "standardization_compensated": standardized,
            "class_direction": "class_0_to_class_1",
            "raw_l2_norm": raw_norm,
            "normalized_l2_norm": float(normal_tensor.norm().item()),
        }

    def _load_dataset(
        self,
        block_index: int,
        step_index: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        block_dir = self.dataset_dir / f"block_{block_index:02d}"

        prefix = f"step_{step_index:02d}"

        features_path = block_dir / f"{prefix}_features.pt"

        labels_path = block_dir / f"{prefix}_labels.pt"

        samples_path = block_dir / f"{prefix}_samples.yaml"

        for path in (
            features_path,
            labels_path,
            samples_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        features_tensor = self._torch_load(features_path)

        labels_tensor = self._torch_load(labels_path)

        if not isinstance(
            features_tensor,
            torch.Tensor,
        ):
            raise TypeError(f"Expected Tensor in {features_path}.")

        if not isinstance(
            labels_tensor,
            torch.Tensor,
        ):
            raise TypeError(f"Expected Tensor in {labels_path}.")

        features = features_tensor.detach().cpu().float().numpy()

        labels = labels_tensor.detach().cpu().long().numpy()

        sample_document = OmegaConf.load(samples_path)

        samples = OmegaConf.to_container(
            sample_document.samples,
            resolve=True,
        )

        if not isinstance(samples, list):
            raise TypeError("samples.yaml must contain a samples list.")

        if features.ndim != 2:
            raise RuntimeError(
                "Expected features shape " "[samples, channels], got " f"{features.shape}."
            )

        if labels.ndim != 1:
            raise RuntimeError(f"Expected labels shape [samples], got {labels.shape}.")

        if features.shape[0] != labels.shape[0] or features.shape[0] != len(samples):
            raise RuntimeError("Features, labels and sample metadata " "have different lengths.")

        if not np.isfinite(features).all():
            raise RuntimeError("Features contain NaN or Inf.")

        if set(np.unique(labels).tolist()) != {
            0,
            1,
        }:
            raise RuntimeError(f"Expected labels 0 and 1, got " f"{np.unique(labels).tolist()}.")

        return features, labels, samples

    @staticmethod
    def _torch_load(
        path: Path,
    ) -> Any:
        try:
            return torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            return torch.load(
                path,
                map_location="cpu",
            )

    @staticmethod
    def _validate_pair_groups(
        labels: np.ndarray,
        samples: list[dict[str, Any]],
    ) -> None:
        labels_by_pair: dict[
            str,
            list[int],
        ] = defaultdict(list)

        roles_by_pair: dict[
            str,
            set[str],
        ] = defaultdict(set)

        for row_index, sample in enumerate(samples):
            pair_name = str(sample["pair_name"])

            role = str(sample["prompt_role"])

            sample_label = int(sample["label"])

            if sample_label != int(labels[row_index]):
                raise RuntimeError(
                    "Label mismatch between tensor and " f"sample metadata at row {row_index}."
                )

            labels_by_pair[pair_name].append(sample_label)

            roles_by_pair[pair_name].add(role)

        for pair_name in labels_by_pair:
            pair_labels = sorted(labels_by_pair[pair_name])

            pair_roles = roles_by_pair[pair_name]

            if pair_labels != [0, 1]:
                raise RuntimeError(
                    f"Pair {pair_name!r} does not have " f"exactly labels [0, 1]: {pair_labels}"
                )

            if pair_roles != {
                "negative",
                "positive",
            }:
                raise RuntimeError(
                    f"Pair {pair_name!r} has invalid " f"roles: {sorted(pair_roles)}"
                )

    @staticmethod
    def _evaluate(
        model: Pipeline,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, Any]:
        predictions = model.predict(features)

        all_probabilities = model.predict_proba(features)

        classes = model.classes_.tolist()

        try:
            positive_index = classes.index(1)
        except ValueError as error:
            raise RuntimeError(f"Positive class 1 is missing: {classes}") from error

        probabilities = all_probabilities[
            :,
            positive_index,
        ]

        decision_scores = model.decision_function(features)

        positive_mask = labels == 1
        negative_mask = labels == 0

        positive_probability_mean = float(probabilities[positive_mask].mean())

        negative_probability_mean = float(probabilities[negative_mask].mean())

        positive_decision_mean = float(decision_scores[positive_mask].mean())

        negative_decision_mean = float(decision_scores[negative_mask].mean())

        return {
            "accuracy": float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    labels,
                    predictions,
                )
            ),
            "roc_auc": float(
                roc_auc_score(
                    labels,
                    probabilities,
                )
            ),
            "log_loss": float(
                log_loss(
                    labels,
                    all_probabilities,
                    labels=[0, 1],
                )
            ),
            "brier_score": float(
                brier_score_loss(
                    labels,
                    probabilities,
                )
            ),
            "confusion_matrix": (
                confusion_matrix(
                    labels,
                    predictions,
                    labels=[0, 1],
                ).tolist()
            ),
            "positive_probability_mean": (positive_probability_mean),
            "negative_probability_mean": (negative_probability_mean),
            "probability_gap": float(positive_probability_mean - negative_probability_mean),
            "positive_decision_mean": (positive_decision_mean),
            "negative_decision_mean": (negative_decision_mean),
            "decision_gap": float(positive_decision_mean - negative_decision_mean),
            "probability_min": float(probabilities.min()),
            "probability_mean": float(probabilities.mean()),
            "probability_max": float(probabilities.max()),
        }
