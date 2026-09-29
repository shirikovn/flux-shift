from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


LocationKey = tuple[int, int]
PairLocationKey = tuple[str, int, int]
DEFAULT_ACTIVATION_LOCATION = "transformer_block_output_text"


class MeanDifferenceCollector:
    """
    Streaming collector for token-wise mean differences.

    For every pair and every selected (block, step):

        difference = positive - negative

    It stores only:
        - one pending negative tensor per pair/location;
        - accumulated difference sums;
        - pair counts.

    Full positive and negative datasets are not retained.
    """

    def __init__(
        self,
        save_dir: str,
        tensor_dtype: str = "float32",
        normalize: bool = True,
        eps: float = 1.0e-8,
        concept_name: str | None = None,
        activation_location: str = DEFAULT_ACTIVATION_LOCATION,
    ) -> None:
        if tensor_dtype not in DTYPE_MAP:
            raise ValueError(
                f"Unsupported tensor_dtype=" f"{tensor_dtype!r}. " f"Available: {list(DTYPE_MAP)}"
            )

        self.save_dir = Path(save_dir)
        self.tensor_dtype = DTYPE_MAP[tensor_dtype]
        self.normalize = bool(normalize)
        self.eps = float(eps)
        self.concept_name = concept_name
        self.activation_location = str(activation_location)

        self._pending_negatives: dict[
            PairLocationKey,
            torch.Tensor,
        ] = {}

        self._difference_sums: dict[
            LocationKey,
            torch.Tensor,
        ] = {}

        # Sum of per-pair, per-token unit differences. Dividing this by the
        # pair count produces a token-wise direction whose norm is also a
        # directional-consistency score in [0, 1]. Unlike normalizing only
        # after averaging, contradictory prompt-pair directions cancel.
        self._unit_difference_sums: dict[
            LocationKey,
            torch.Tensor,
        ] = {}

        self._pair_counts: dict[
            LocationKey,
            int,
        ] = {}

        self._pair_names: dict[
            LocationKey,
            list[str],
        ] = {}

        self._timesteps: dict[
            LocationKey,
            float | None,
        ] = {}

    def add(
        self,
        pair_name: str,
        prompt_role: str,
        block_index: int,
        step_index: int,
        timestep: float | None,
        activation: torch.Tensor,
    ) -> None:
        tensor = (
            activation.detach()
            .to(
                device="cpu",
                dtype=self.tensor_dtype,
            )
            .clone()
        )

        self._validate_activation(
            tensor=tensor,
            pair_name=pair_name,
            block_index=block_index,
            step_index=step_index,
        )

        location_key: LocationKey = (
            block_index,
            step_index,
        )

        pair_location_key: PairLocationKey = (
            pair_name,
            block_index,
            step_index,
        )

        if prompt_role == "negative":
            self._add_negative(
                pair_location_key=pair_location_key,
                tensor=tensor,
            )

        elif prompt_role == "positive":
            self._add_positive(
                pair_name=pair_name,
                location_key=location_key,
                pair_location_key=(pair_location_key),
                tensor=tensor,
                timestep=timestep,
            )

        else:
            raise ValueError(f"Unknown prompt role: " f"{prompt_role!r}")

    def _add_negative(
        self,
        pair_location_key: PairLocationKey,
        tensor: torch.Tensor,
    ) -> None:
        if pair_location_key in self._pending_negatives:
            raise RuntimeError("Duplicate negative activation for " f"{pair_location_key}.")

        self._pending_negatives[pair_location_key] = tensor

    def _add_positive(
        self,
        pair_name: str,
        location_key: LocationKey,
        pair_location_key: PairLocationKey,
        tensor: torch.Tensor,
        timestep: float | None,
    ) -> None:
        negative = self._pending_negatives.pop(
            pair_location_key,
            None,
        )

        if negative is None:
            raise RuntimeError(
                "Positive activation was received "
                "without a corresponding negative "
                f"activation: {pair_location_key}."
            )

        if negative.shape != tensor.shape:
            raise RuntimeError(
                "Positive and negative activation "
                "shapes differ: "
                f"{negative.shape} != {tensor.shape}."
            )

        difference = tensor - negative

        if not torch.isfinite(difference).all():
            raise RuntimeError("The activation difference contains " "NaN or Inf values.")

        if location_key not in self._difference_sums:
            self._difference_sums[location_key] = torch.zeros_like(difference)

            self._unit_difference_sums[location_key] = torch.zeros_like(
                difference
            )

            self._pair_counts[location_key] = 0

            self._pair_names[location_key] = []

            self._timesteps[location_key] = timestep

        self._difference_sums[location_key].add_(difference)

        difference_norms = difference.norm(dim=-1, keepdim=True)
        unit_difference = difference / difference_norms.clamp_min(self.eps)
        unit_difference = torch.where(
            difference_norms > self.eps,
            unit_difference,
            torch.zeros_like(unit_difference),
        )
        self._unit_difference_sums[location_key].add_(unit_difference)

        self._pair_counts[location_key] += 1

        self._pair_names[location_key].append(pair_name)

    @staticmethod
    def _validate_activation(
        tensor: torch.Tensor,
        pair_name: str,
        block_index: int,
        step_index: int,
    ) -> None:
        if tensor.ndim != 3:
            raise RuntimeError(
                "Expected activation shape "
                "[batch, tokens, channels], got "
                f"{tuple(tensor.shape)} for "
                f"pair={pair_name!r}, "
                f"block={block_index}, "
                f"step={step_index}."
            )

        if tensor.shape[0] != 1:
            raise RuntimeError(
                "Stage 3 currently expects " "batch_size=1, got " f"{tensor.shape[0]}."
            )

        if not torch.isfinite(tensor).all():
            raise RuntimeError("Activation contains NaN or Inf.")

    def save(self) -> Path:
        if self._pending_negatives:
            missing = sorted(self._pending_negatives)
            raise RuntimeError(
                "Some negative activations have no matching " f"positive activations: {missing}"
            )

        if not self._difference_sums:
            raise RuntimeError("No activation differences were collected.")

        self.save_dir.mkdir(parents=True, exist_ok=True)

        location_metadata: list[dict[str, Any]] = []

        for location_key in sorted(self._difference_sums):
            block_index, step_index = location_key
            count = self._pair_counts[location_key]

            if count <= 0:
                raise RuntimeError(f"Invalid pair count for {location_key}: {count}")

            mean_difference = self._difference_sums[location_key] / count

            # Shape: [tokens, channels].
            tokenwise_raw = mean_difference.squeeze(0)
            tokenwise_vector = self._normalize(tokenwise_raw)

            # Shape: [tokens, channels]. This remains token-wise: each prompt
            # pair is normalized along channels before the prompt-pair mean.
            # The resulting token norm measures agreement across pairs and is
            # intentionally not normalized away.
            consistent_vector = (
                self._unit_difference_sums[location_key] / count
            ).squeeze(0)

            # Shape: [channels].
            #
            # This is the prompt-pair mean difference followed by
            # averaging over text-token positions.
            token_mean_raw = tokenwise_raw.mean(dim=0)
            token_mean_vector = self._normalize(token_mean_raw)

            block_dir = self.save_dir / f"block_{block_index:02d}"
            block_dir.mkdir(parents=True, exist_ok=True)

            prefix = f"step_{step_index:02d}"

            # Keep the original filenames for backward compatibility.
            artifacts = {
                "raw_difference": tokenwise_raw,
                "vector": tokenwise_vector,
                "consistent_vector": consistent_vector,
                "token_mean_raw_difference": token_mean_raw,
                "token_mean_vector": token_mean_vector,
            }

            artifact_paths: dict[str, Path] = {}

            for artifact_name, tensor in artifacts.items():
                path = block_dir / f"{prefix}_{artifact_name}.pt"
                torch.save(tensor, path)
                artifact_paths[artifact_name] = path

            token_norms = tokenwise_vector.norm(dim=-1)
            consistency = consistent_vector.norm(dim=-1)

            location_metadata.append(
                {
                    "block_index": block_index,
                    "step_index": step_index,
                    "timestep": self._timesteps[location_key],
                    "num_prompt_pairs": count,
                    "pair_names": self._pair_names[location_key],
                    "tokenwise_difference": {
                        "raw_shape": list(tokenwise_raw.shape),
                        "vector_shape": list(tokenwise_vector.shape),
                        "raw_l2_norm": float(tokenwise_raw.norm().item()),
                        "token_norm_min": float(token_norms.min().item()),
                        "token_norm_mean": float(token_norms.mean().item()),
                        "token_norm_max": float(token_norms.max().item()),
                        "raw_path": str(artifact_paths["raw_difference"]),
                        "vector_path": str(artifact_paths["vector"]),
                    },
                    "token_mean_difference": {
                        "raw_shape": list(token_mean_raw.shape),
                        "vector_shape": list(token_mean_vector.shape),
                        "raw_l2_norm": float(token_mean_raw.norm().item()),
                        "vector_l2_norm": float(token_mean_vector.norm().item()),
                        "raw_path": str(artifact_paths["token_mean_raw_difference"]),
                        "vector_path": str(artifact_paths["token_mean_vector"]),
                    },
                    "tokenwise_consistent_difference": {
                        "vector_shape": list(consistent_vector.shape),
                        "estimator": (
                            "mean_over_pairs_of_channel_l2_normalized_"
                            "pair_differences"
                        ),
                        "token_consistency_min": float(consistency.min().item()),
                        "token_consistency_mean": float(consistency.mean().item()),
                        "token_consistency_max": float(consistency.max().item()),
                        "token_fraction_ge_0_25": float(
                            (consistency >= 0.25).float().mean().item()
                        ),
                        "token_fraction_ge_0_50": float(
                            (consistency >= 0.50).float().mean().item()
                        ),
                        "token_fraction_ge_0_75": float(
                            (consistency >= 0.75).float().mean().item()
                        ),
                        "vector_path": str(
                            artifact_paths["consistent_vector"]
                        ),
                    },
                }
            )

        normalization = "channel_l2" if self.normalize else "none"

        metadata = {
            "concept_name": self.concept_name,
            "activation_location": self.activation_location,
            "difference_direction": "positive_minus_negative",
            "tensor_dtype": str(self.tensor_dtype),
            "eps": self.eps,
            "vector_types": {
                "tokenwise_difference": {
                    "shape": "[tokens, channels]",
                    "normalization": normalization,
                    "filename": "step_XX_vector.pt",
                },
                "token_mean_difference": {
                    "shape": "[channels]",
                    "pooling": "mean_over_text_tokens",
                    "normalization": normalization,
                    "filename": "step_XX_token_mean_vector.pt",
                },
                "tokenwise_consistent_difference": {
                    "shape": "[tokens, channels]",
                    "pair_normalization": "channel_l2_before_pair_mean",
                    "final_normalization": "none",
                    "token_norm_interpretation": (
                        "directional_consistency_across_prompt_pairs"
                    ),
                    "filename": "step_XX_consistent_vector.pt",
                },
            },
            "locations": location_metadata,
        }

        metadata_path = self.save_dir / "metadata.yaml"

        OmegaConf.save(
            config=OmegaConf.create(metadata),
            f=metadata_path,
        )

        return metadata_path

    def _normalize(
        self,
        raw_vector: torch.Tensor,
    ) -> torch.Tensor:
        if not self.normalize:
            return raw_vector.clone()

        norms = raw_vector.norm(
            dim=-1,
            keepdim=True,
        )

        vector = raw_vector / norms.clamp_min(self.eps)

        # Truly zero vectors should remain zero.
        vector = torch.where(
            norms > self.eps,
            vector,
            torch.zeros_like(vector),
        )

        return vector

    def summary(self) -> dict[str, Any]:
        return {
            "type": (self.__class__.__name__),
            "save_dir": str(self.save_dir),
            "concept_name": (self.concept_name),
            "activation_location": self.activation_location,
            "tensor_dtype": str(self.tensor_dtype),
            "normalize": self.normalize,
            "num_locations": len(self._difference_sums),
            "pair_counts": {
                (f"block_{block:02d}" f"_step_{step:02d}"): count
                for (
                    block,
                    step,
                ), count in sorted(self._pair_counts.items())
            },
        }


class PooledSVMDatasetCollector:
    """
    Collects token-averaged DiT text activations for linear SVM training.

    Input activation:
        [batch, tokens, channels]

    Stored classifier feature:
        [channels]

    Each selected block/step receives one feature for every
    negative and positive prompt.
    """

    ROLE_TO_LABEL: dict[str, int] = {
        "negative": 0,
        "positive": 1,
    }

    def __init__(
        self,
        save_dir: str,
        tensor_dtype: str = "float32",
        concept_name: str | None = None,
        activation_location: str = DEFAULT_ACTIVATION_LOCATION,
    ) -> None:
        if tensor_dtype not in DTYPE_MAP:
            raise ValueError(
                f"Unsupported tensor_dtype={tensor_dtype!r}. " f"Available: {sorted(DTYPE_MAP)}"
            )

        self.save_dir = Path(save_dir)
        self.tensor_dtype = DTYPE_MAP[tensor_dtype]
        self.concept_name = concept_name
        self.activation_location = str(activation_location)

        # Key: (block_index, step_index)
        self._features: dict[
            tuple[int, int],
            list[torch.Tensor],
        ] = {}

        self._samples: dict[
            tuple[int, int],
            list[dict[str, Any]],
        ] = {}

        # Prevent accidental duplicate captures.
        self._seen: set[tuple[str, str, int, int]] = set()

    def add(
        self,
        pair_name: str,
        prompt_role: str,
        block_index: int,
        step_index: int,
        timestep: float | None,
        activation: torch.Tensor,
    ) -> None:
        if prompt_role not in self.ROLE_TO_LABEL:
            raise ValueError(f"Unknown prompt_role={prompt_role!r}.")

        self._validate_activation(
            activation=activation,
            pair_name=pair_name,
            block_index=block_index,
            step_index=step_index,
        )

        sample_key = (
            pair_name,
            prompt_role,
            block_index,
            step_index,
        )

        if sample_key in self._seen:
            raise RuntimeError(f"Duplicate SVM sample: {sample_key}")

        self._seen.add(sample_key)

        # Convert before mean so accumulation is performed in float32
        # in the recommended configuration.
        cpu_activation = activation.detach().to(
            device="cpu",
            dtype=self.tensor_dtype,
        )

        pooled = cpu_activation.mean(dim=1).squeeze(0).clone()

        if pooled.ndim != 1:
            raise RuntimeError(
                "Expected pooled activation shape [channels], got " f"{tuple(pooled.shape)}."
            )

        if not torch.isfinite(pooled).all():
            raise RuntimeError("Pooled activation contains NaN or Inf.")

        location = (
            int(block_index),
            int(step_index),
        )

        features = self._features.setdefault(
            location,
            [],
        )

        samples = self._samples.setdefault(
            location,
            [],
        )

        row_index = len(features)
        label = self.ROLE_TO_LABEL[prompt_role]

        features.append(pooled)

        samples.append(
            {
                "row_index": row_index,
                "pair_name": pair_name,
                "prompt_role": prompt_role,
                "label": label,
                "block_index": block_index,
                "step_index": step_index,
                "timestep": timestep,
            }
        )

    @staticmethod
    def _validate_activation(
        activation: torch.Tensor,
        pair_name: str,
        block_index: int,
        step_index: int,
    ) -> None:
        if not isinstance(activation, torch.Tensor):
            raise TypeError(f"Expected Tensor, got {type(activation)!r}.")

        if activation.ndim != 3:
            raise RuntimeError(
                "Expected activation shape "
                "[batch, tokens, channels], got "
                f"{tuple(activation.shape)} for "
                f"pair={pair_name!r}, "
                f"block={block_index}, "
                f"step={step_index}."
            )

        if activation.shape[0] != 1:
            raise RuntimeError(
                "SVM dataset collection currently expects "
                f"batch_size=1, got {activation.shape[0]}."
            )

        if not torch.isfinite(activation).all():
            raise RuntimeError("Activation contains NaN or Inf.")

    def save(self) -> Path:
        if not self._features:
            raise RuntimeError("No SVM features were collected.")

        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        locations_metadata: list[dict[str, Any]] = []

        for location in sorted(self._features):
            block_index, step_index = location

            feature_list = self._features[location]
            sample_list = self._samples[location]

            features = torch.stack(
                feature_list,
                dim=0,
            )

            labels = torch.tensor(
                [int(sample["label"]) for sample in sample_list],
                dtype=torch.long,
            )

            if features.ndim != 2:
                raise RuntimeError(
                    "Expected feature matrix [samples, channels], got " f"{tuple(features.shape)}."
                )

            if labels.shape != (features.shape[0],):
                raise RuntimeError(
                    "Feature and label counts differ: " f"{features.shape[0]} != {labels.shape[0]}."
                )

            class_counts = Counter(labels.tolist())

            if set(class_counts) != {0, 1}:
                raise RuntimeError(f"Both classes are required, got {class_counts}.")

            location_dir = self.save_dir / f"block_{block_index:02d}"

            location_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            prefix = f"step_{step_index:02d}"

            features_path = location_dir / f"{prefix}_features.pt"

            labels_path = location_dir / f"{prefix}_labels.pt"

            samples_path = location_dir / f"{prefix}_samples.yaml"

            torch.save(
                features,
                features_path,
            )

            torch.save(
                labels,
                labels_path,
            )

            OmegaConf.save(
                config=OmegaConf.create(
                    {
                        "samples": sample_list,
                    }
                ),
                f=samples_path,
            )

            feature_norms = features.norm(dim=-1)

            pair_names = sorted({str(sample["pair_name"]) for sample in sample_list})

            locations_metadata.append(
                {
                    "block_index": block_index,
                    "step_index": step_index,
                    "num_samples": int(features.shape[0]),
                    "num_features": int(features.shape[1]),
                    "num_pairs": len(pair_names),
                    "pair_names": pair_names,
                    "class_counts": {
                        "negative": int(class_counts[0]),
                        "positive": int(class_counts[1]),
                    },
                    "dtype": str(features.dtype),
                    "feature_mean_abs": float(features.abs().mean().item()),
                    "feature_norm_min": float(feature_norms.min().item()),
                    "feature_norm_mean": float(feature_norms.mean().item()),
                    "feature_norm_max": float(feature_norms.max().item()),
                    "features_path": str(features_path),
                    "labels_path": str(labels_path),
                    "samples_path": str(samples_path),
                }
            )

        metadata = {
            "concept_name": self.concept_name,
            "activation_location": self.activation_location,
            "dataset_type": ("token_averaged_dit_activations"),
            "pooling": "mean_over_text_tokens",
            "tensor_dtype": str(self.tensor_dtype),
            "num_locations": len(locations_metadata),
            "locations": locations_metadata,
        }

        metadata_path = self.save_dir / "metadata.yaml"

        OmegaConf.save(
            config=OmegaConf.create(metadata),
            f=metadata_path,
        )

        return metadata_path

    def summary(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "save_dir": str(self.save_dir),
            "concept_name": self.concept_name,
            "activation_location": self.activation_location,
            "num_locations": len(self._features),
            "sample_counts": {
                (f"block_{block:02d}" f"_step_{step:02d}"): len(features)
                for (
                    block,
                    step,
                ), features in sorted(self._features.items())
            },
        }


class CombinedDiTCollector:
    """
    Sends each captured DiT activation to two collectors:

    1. MeanDifferenceCollector:
       builds block-specific token-wise steering vectors.

    2. PooledSVMDatasetCollector:
       mean-pools the same activation over tokens and builds
       a dataset for block-specific SVM classifiers.

    The FLUX generation and hook call therefore happen only once.
    """

    def __init__(
        self,
        save_dir: str,
        tensor_dtype: str = "float32",
        concept_name: str | None = None,
        normalize: bool = True,
        eps: float = 1.0e-8,
        activation_location: str = DEFAULT_ACTIVATION_LOCATION,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.tensor_dtype = str(tensor_dtype)
        self.concept_name = concept_name
        self.normalize = bool(normalize)
        self.eps = float(eps)
        self.activation_location = str(activation_location)

        if self.eps <= 0:
            raise ValueError("eps must be positive.")

        # Child collectors are constructed here rather than
        # through nested Hydra configs. This keeps the
        # intervention config small and avoids recursive
        # instantiation issues.
        self.vector_collector = MeanDifferenceCollector(
            save_dir=str(self.save_dir / "vectors"),
            tensor_dtype=self.tensor_dtype,
            concept_name=self.concept_name,
            normalize=self.normalize,
            eps=self.eps,
            activation_location=self.activation_location,
        )

        self.svm_collector = PooledSVMDatasetCollector(
            save_dir=str(self.save_dir / "svm_dataset"),
            tensor_dtype=self.tensor_dtype,
            concept_name=self.concept_name,
            activation_location=self.activation_location,
        )

        self.total_add_calls = 0
        self.calls_by_location: dict[
            str,
            int,
        ] = {}

    def add(
        self,
        pair_name: str,
        prompt_role: str,
        block_index: int,
        step_index: int,
        timestep: float | None,
        activation: Any,
    ) -> None:
        """
        Fan out one hook activation to both collectors.

        No second FLUX generation is needed.
        """
        location = f"block_{int(block_index):02d}" f"_step_{int(step_index):02d}"

        # Both collectors receive the same activation.
        #
        # MeanDifferenceCollector retains token-wise structure.
        # PooledSVMDatasetCollector performs mean(dim=1).
        self.vector_collector.add(
            pair_name=pair_name,
            prompt_role=prompt_role,
            block_index=block_index,
            step_index=step_index,
            timestep=timestep,
            activation=activation,
        )

        self.svm_collector.add(
            pair_name=pair_name,
            prompt_role=prompt_role,
            block_index=block_index,
            step_index=step_index,
            timestep=timestep,
            activation=activation,
        )

        self.total_add_calls += 1

        self.calls_by_location[location] = (
            self.calls_by_location.get(
                location,
                0,
            )
            + 1
        )

    def save(self) -> Path:
        """
        Save both artifact groups and one combined metadata file.

        Returns a Path for compatibility with the existing
        activation collection pipeline.
        """
        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        vectors_metadata_path = self.vector_collector.save()

        svm_metadata_path = self.svm_collector.save()

        combined_metadata = {
            "collector_type": (self.__class__.__name__),
            "concept_name": (self.concept_name),
            "activation_location": self.activation_location,
            "tensor_dtype": (self.tensor_dtype),
            "normalize_vectors": (self.normalize),
            "eps": self.eps,
            "total_add_calls": (self.total_add_calls),
            "calls_by_location": dict(self.calls_by_location),
            "artifacts": {
                "vectors_directory": str(self.save_dir / "vectors"),
                "vectors_metadata": str(vectors_metadata_path),
                "svm_dataset_directory": str(self.save_dir / "svm_dataset"),
                "svm_dataset_metadata": str(svm_metadata_path),
            },
            "collectors": {
                "vectors": (self.vector_collector.summary()),
                "svm_dataset": (self.svm_collector.summary()),
            },
        }

        metadata_path = self.save_dir / "metadata.yaml"

        OmegaConf.save(
            config=OmegaConf.create(combined_metadata),
            f=metadata_path,
        )

        return metadata_path

    def summary(self) -> dict[str, Any]:
        return {
            "type": (self.__class__.__name__),
            "save_dir": str(self.save_dir),
            "concept_name": (self.concept_name),
            "tensor_dtype": (self.tensor_dtype),
            "normalize_vectors": (self.normalize),
            "total_add_calls": (self.total_add_calls),
            "calls_by_location": dict(self.calls_by_location),
            "vectors": (self.vector_collector.summary()),
            "svm_dataset": (self.svm_collector.summary()),
        }
