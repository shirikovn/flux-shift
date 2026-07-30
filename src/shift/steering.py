from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


class TokenWiseSteeringController:
    """
    Applies block-specific token-wise steering vectors.

    Static mode:
        effective_strength = strength

    Dynamic mode:
        effective_strength = strength * eta_cls
    """

    OPERATION_SIGNS: dict[str, float] = {
        "add": 1.0,
        "erase": -1.0,
    }

    def __init__(
        self,
        vector_paths: Mapping[Any, str] | None = None,
        vector_directory: str | None = None,
        block_indices: Sequence[int] | None = None,
        source_step: int = 0,
        operation: str = "erase",
        strength: float = 0.0,
        regularizer: Any | None = None,
        use_classifier: bool = False,
        validate_runtime: bool = False,
    ) -> None:
        resolved_paths = self._resolve_vector_paths(
            vector_paths=vector_paths,
            vector_directory=vector_directory,
            block_indices=block_indices,
            source_step=source_step,
        )

        self.source_step = int(source_step)
        self.regularizer = regularizer
        self.validate_runtime = bool(
            validate_runtime
        )

        self._cpu_vectors: dict[
            int,
            torch.Tensor,
        ] = {}

        self._vector_paths: dict[
            int,
            str,
        ] = {}

        for block_index, path in sorted(
            resolved_paths.items()
        ):
            vector = self._load_vector(
                block_index=block_index,
                path=path,
            )

            self._cpu_vectors[
                block_index
            ] = vector

            self._vector_paths[
                block_index
            ] = str(path)

        self._runtime_cache: dict[
            tuple[int, str, torch.dtype],
            torch.Tensor,
        ] = {}

        self.operation = "erase"
        self.strength = 0.0
        self.use_classifier = False

        self.reset_statistics()

        self.configure(
            operation=operation,
            strength=strength,
            use_classifier=use_classifier,
        )

    @staticmethod
    def _resolve_vector_paths(
        vector_paths: Mapping[Any, str] | None,
        vector_directory: str | None,
        block_indices: Sequence[int] | None,
        source_step: int,
    ) -> dict[int, Path]:
        explicit_paths = (
            vector_paths is not None
            and len(vector_paths) > 0
        )

        directory_configuration = (
            vector_directory is not None
            or block_indices is not None
        )

        if (
            explicit_paths
            and directory_configuration
        ):
            raise ValueError(
                "Specify either vector_paths or "
                "vector_directory + block_indices."
            )

        if explicit_paths:
            return {
                int(raw_block): Path(
                    str(raw_path)
                )
                for raw_block, raw_path
                in vector_paths.items()
            }

        if vector_directory is None:
            raise ValueError(
                "vector_directory is required."
            )

        if block_indices is None:
            raise ValueError(
                "block_indices is required."
            )

        root = Path(vector_directory)

        return {
            int(block_index): (
                root
                / f"block_{int(block_index):02d}"
                / (
                    f"step_{int(source_step):02d}"
                    "_vector.pt"
                )
            )
            for block_index in block_indices
        }

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

    def _load_vector(
        self,
        block_index: int,
        path: Path,
    ) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(
                f"Vector for block {block_index} "
                f"does not exist: {path}"
            )

        value = self._torch_load(path)

        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise TypeError(
                f"Expected tensor in {path}."
            )

        if value.ndim != 2:
            raise RuntimeError(
                "Expected vector shape "
                "[tokens, channels], got "
                f"{tuple(value.shape)}."
            )

        if not torch.isfinite(value).all():
            raise RuntimeError(
                f"Vector in {path} contains "
                "NaN or Inf."
            )

        return (
            value.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .contiguous()
        )

    @property
    def available_blocks(self) -> set[int]:
        return set(self._cpu_vectors)

    def configure(
        self,
        operation: str,
        strength: float,
        use_classifier: bool | None = None,
    ) -> None:
        if operation not in self.OPERATION_SIGNS:
            raise ValueError(
                f"Unsupported operation={operation!r}."
            )

        strength = float(strength)

        if strength < 0:
            raise ValueError(
                "Strength must be nonnegative."
            )

        next_use_classifier = (
            self.use_classifier
            if use_classifier is None
            else bool(use_classifier)
        )

        if (
            next_use_classifier
            and self.regularizer is None
        ):
            raise ValueError(
                "use_classifier=True requires "
                "a regularizer."
            )

        # For this reproduction stage we use the
        # classifier only for concept erasure.
        #
        # For addition, p_cls is initially small,
        # so the paper's erasure-oriented formula
        # would suppress the desired intervention.
        if (
            next_use_classifier
            and operation != "erase"
        ):
            raise ValueError(
                "Dynamic SVM regularization is "
                "currently supported only for "
                "operation='erase'."
            )

        self.operation = operation
        self.strength = strength
        self.use_classifier = (
            next_use_classifier
        )

    def apply(
        self,
        block_index: int,
        step_index: int,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        self.total_calls += 1

        if self.strength == 0.0:
            return activation

        if block_index not in self._cpu_vectors:
            raise KeyError(
                f"No vector loaded for block "
                f"{block_index}."
            )

        if activation.ndim != 3:
            raise RuntimeError(
                "Expected activation shape "
                "[batch, tokens, channels], got "
                f"{tuple(activation.shape)}."
            )

        location = (
            f"block_{block_index:02d}"
            f"_step_{step_index:02d}"
        )

        p_cls: float | None = None
        eta_cls: float | None = None

        effective_strength = self.strength

        if self.use_classifier:
            p_cls, eta_cls = (
                self.regularizer.predict(
                    block_index=block_index,
                    activation=activation,
                )
            )

            effective_strength *= eta_cls

            self.classifier_by_location[
                location
            ] = {
                "p_cls": p_cls,
                "eta_cls": eta_cls,
                "base_strength": (
                    self.strength
                ),
                "effective_strength": (
                    effective_strength
                ),
            }

        if effective_strength == 0.0:
            return activation

        vector = self._get_runtime_vector(
            block_index=block_index,
            activation=activation,
        )

        expected_shape = tuple(
            activation.shape[1:]
        )

        if tuple(vector.shape) != expected_shape:
            raise RuntimeError(
                "Vector and activation shape "
                "mismatch: "
                f"vector={tuple(vector.shape)}, "
                f"activation={tuple(activation.shape)}."
            )

        sign = self.OPERATION_SIGNS[
            self.operation
        ]

        scale = (
            sign
            * effective_strength
        )

        steered = (
            activation
            + scale * vector.unsqueeze(0)
        )

        if (
            self.validate_runtime
            and not torch.isfinite(steered).all()
        ):
            raise RuntimeError(
                "Steered activation contains "
                "NaN or Inf."
            )

        self.modified_calls += 1

        self.calls_by_location[location] = (
            self.calls_by_location.get(
                location,
                0,
            )
            + 1
        )

        relative_scale = (
            self._estimate_relative_scale(
                activation=activation,
                vector=vector,
                effective_strength=(
                    effective_strength
                ),
            )
        )

        self.relative_scale_by_location[
            location
        ] = relative_scale

        self.effective_strength_by_location[
            location
        ] = effective_strength

        return steered

    @staticmethod
    def _estimate_relative_scale(
        activation: torch.Tensor,
        vector: torch.Tensor,
        effective_strength: float,
    ) -> float:
        activation_norm = (
            activation.detach()
            .float()
            .norm(dim=-1)
            .mean()
            .item()
        )

        vector_norm = (
            vector.detach()
            .float()
            .norm(dim=-1)
            .mean()
            .item()
        )

        delta_norm = (
            abs(effective_strength)
            * vector_norm
        )

        return float(
            delta_norm
            / max(activation_norm, 1.0e-8)
        )

    def _get_runtime_vector(
        self,
        block_index: int,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        cache_key = (
            block_index,
            str(activation.device),
            activation.dtype,
        )

        cached = self._runtime_cache.get(
            cache_key
        )

        if cached is not None:
            return cached

        vector = self._cpu_vectors[
            block_index
        ].to(
            device=activation.device,
            dtype=activation.dtype,
            non_blocking=True,
        )

        self._runtime_cache[
            cache_key
        ] = vector

        return vector

    def reset_statistics(self) -> None:
        self.total_calls = 0
        self.modified_calls = 0

        self.calls_by_location: dict[
            str,
            int,
        ] = {}

        self.relative_scale_by_location: dict[
            str,
            float,
        ] = {}

        self.effective_strength_by_location: dict[
            str,
            float,
        ] = {}

        self.classifier_by_location: dict[
            str,
            dict[str, float],
        ] = {}

    def clear_runtime_cache(self) -> None:
        self._runtime_cache.clear()

    @staticmethod
    def _summary_values(
        values: list[float],
    ) -> dict[str, float | None]:
        if not values:
            return {
                "min": None,
                "mean": None,
                "max": None,
            }

        return {
            "min": min(values),
            "mean": (
                sum(values)
                / len(values)
            ),
            "max": max(values),
        }

    def statistics(self) -> dict[str, Any]:
        relative_scales = list(
            self.relative_scale_by_location.values()
        )

        effective_strengths = list(
            self.effective_strength_by_location.values()
        )

        probabilities = [
            record["p_cls"]
            for record
            in self.classifier_by_location.values()
        ]

        eta_values = [
            record["eta_cls"]
            for record
            in self.classifier_by_location.values()
        ]

        return {
            "operation": self.operation,
            "base_strength": self.strength,
            "use_classifier": (
                self.use_classifier
            ),
            "total_calls": self.total_calls,
            "modified_calls": (
                self.modified_calls
            ),
            "calls_by_location": dict(
                self.calls_by_location
            ),
            "classifier_by_location": dict(
                self.classifier_by_location
            ),
            "effective_strength_by_location": dict(
                self.effective_strength_by_location
            ),
            "relative_scale_by_location": dict(
                self.relative_scale_by_location
            ),
            "p_cls": self._summary_values(
                probabilities
            ),
            "eta_cls": self._summary_values(
                eta_values
            ),
            "effective_strength": (
                self._summary_values(
                    effective_strengths
                )
            ),
            "relative_scale": (
                self._summary_values(
                    relative_scales
                )
            ),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "source_step": self.source_step,
            "available_blocks": sorted(
                self.available_blocks
            ),
            "statistics": self.statistics(),
            "regularizer": (
                self.regularizer.summary()
                if self.regularizer is not None
                else None
            ),
            "vectors": {
                str(block_index): {
                    "path": (
                        self._vector_paths[
                            block_index
                        ]
                    ),
                    "shape": list(
                        self._cpu_vectors[
                            block_index
                        ].shape
                    ),
                }
                for block_index in sorted(
                    self._cpu_vectors
                )
            },
        }
