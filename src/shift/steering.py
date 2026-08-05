from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from src.shift.vector_store import (
    SteeringVectorStore,
)


class TokenWiseSteeringController:
    """
    Applies block-specific steering vectors.

    Supported vector types:

    tokenwise_difference:
        Shape [tokens, channels].

    token_mean_difference:
        Shape [channels], broadcast across tokens.

    svm_normal:
        Shape [channels], broadcast across tokens.

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
        vector_paths: Mapping[Any, Any] | None = None,
        vector_directory: str | None = None,
        svm_normal_directory: str | None = None,
        block_indices: Sequence[int] | None = None,
        step_indices: Sequence[int] | None = None,
        source_step: int = 0,
        timing_mode: str = "shared",
        vector_type: str = "tokenwise_difference",
        operation: str = "erase",
        strength: float = 0.0,
        regularizer: Any | None = None,
        use_classifier: bool = False,
        validate_runtime: bool = False,
    ) -> None:
        self.vector_store = SteeringVectorStore(
            vector_type=vector_type,
            timing_mode=timing_mode,
            vector_paths=vector_paths,
            vector_directory=vector_directory,
            svm_normal_directory=(svm_normal_directory),
            block_indices=block_indices,
            step_indices=step_indices,
            source_step=source_step,
        )

        self.vector_type = self.vector_store.vector_type

        self.timing_mode = self.vector_store.timing_mode

        self.source_step = self.vector_store.source_step

        self.regularizer = regularizer
        self.validate_runtime = bool(validate_runtime)

        self._active_blocks: tuple[int, ...] | None = None
        self._active_steps: tuple[int, ...] | None = None

        self.operation = "erase"
        self.strength = 0.0
        self.use_classifier = False

        self.reset_statistics()

        self.configure(
            operation=operation,
            strength=strength,
            use_classifier=use_classifier,
        )

    @property
    def available_blocks(
        self,
    ) -> set[int]:
        return self.vector_store.available_blocks

    def validate_locations(
        self,
        blocks: Sequence[int] | None,
        steps: Sequence[int] | None,
    ) -> None:
        self.vector_store.validate_locations(
            blocks=blocks,
            steps=steps,
        )

        self._active_blocks = None if blocks is None else tuple(int(value) for value in blocks)

        self._active_steps = None if steps is None else tuple(int(value) for value in steps)

        if self.use_classifier and self.regularizer is not None:
            self.regularizer.validate_locations(
                blocks=blocks,
                steps=steps,
            )

    def vector_configuration(
        self,
    ) -> dict[str, Any]:
        return self.vector_store.configuration()

    def classifier_configuration(
        self,
    ) -> dict[str, Any] | None:
        if self.regularizer is None:
            return None

        return self.regularizer.configuration()

    def configure(
        self,
        operation: str,
        strength: float,
        use_classifier: bool | None = None,
    ) -> None:
        if operation not in self.OPERATION_SIGNS:
            raise ValueError(f"Unsupported operation={operation!r}.")

        strength = float(strength)

        if strength < 0:
            raise ValueError("Strength must be nonnegative.")

        next_use_classifier = (
            self.use_classifier if use_classifier is None else bool(use_classifier)
        )

        if next_use_classifier and self.regularizer is None:
            raise ValueError("use_classifier=True requires a regularizer.")

        if (
            next_use_classifier
            and self.regularizer is not None
            and self._active_blocks is not None
            and self._active_steps is not None
        ):
            self.regularizer.validate_locations(
                blocks=self._active_blocks,
                steps=self._active_steps,
            )

        # For this reproduction stage we use the
        # classifier only for concept erasure.
        #
        # For addition, p_cls is initially small,
        # so the paper's erasure-oriented formula
        # would suppress the desired intervention.
        if next_use_classifier and operation != "erase":
            raise ValueError(
                "Dynamic SVM regularization is "
                "currently supported only for "
                "operation='erase'."
            )

        self.operation = operation
        self.strength = strength
        self.use_classifier = next_use_classifier

    def _validate_vector_shape(
        self,
        vector: torch.Tensor,
        activation: torch.Tensor,
    ) -> None:
        num_tokens = int(activation.shape[1])
        num_channels = int(activation.shape[2])

        if vector.ndim == 2:
            expected_shape = (
                num_tokens,
                num_channels,
            )

            if tuple(vector.shape) != expected_shape:
                raise RuntimeError(
                    "Token-wise vector and activation shape mismatch: "
                    f"vector={tuple(vector.shape)}, "
                    f"expected={expected_shape}, "
                    f"activation={tuple(activation.shape)}."
                )

            return

        if vector.ndim == 1:
            if int(vector.shape[0]) != num_channels:
                raise RuntimeError(
                    "Channel vector and activation shape mismatch: "
                    f"vector={tuple(vector.shape)}, "
                    f"expected=({num_channels},), "
                    f"activation={tuple(activation.shape)}."
                )

            return

        raise RuntimeError(f"Unsupported runtime vector shape: {tuple(vector.shape)}.")

    def apply(
        self,
        block_index: int,
        step_index: int,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        self.total_calls += 1

        if self.strength == 0.0:
            return activation

        if activation.ndim != 3:
            raise RuntimeError(
                "Expected activation shape "
                "[batch, tokens, channels], got "
                f"{tuple(activation.shape)}."
            )

        location = f"block_{block_index:02d}" f"_step_{step_index:02d}"

        p_cls: float | None = None
        eta_cls: float | None = None

        effective_strength = self.strength

        if self.use_classifier:
            (
                p_cls,
                eta_cls,
                classifier_source,
            ) = self.regularizer.predict(
                block_index=block_index,
                step_index=step_index,
                activation=activation,
            )

            effective_strength *= eta_cls

            source_block, source_step = classifier_source

            self.classifier_by_location[location] = {
                "p_cls": p_cls,
                "eta_cls": eta_cls,
                "base_strength": self.strength,
                "effective_strength": effective_strength,
                "source_block": source_block,
                "source_step": source_step,
                "path": self.regularizer.path_for(classifier_source),
            }

        if effective_strength == 0.0:
            return activation

        vector, source_location = self.vector_store.get(
            block_index=block_index,
            runtime_step=step_index,
            device=activation.device,
            dtype=activation.dtype,
        )

        self._validate_vector_shape(
            vector=vector,
            activation=activation,
        )

        sign = self.OPERATION_SIGNS[self.operation]
        scale = sign * effective_strength

        if vector.ndim == 2:
            # [tokens, channels] -> [1, tokens, channels]
            steering_delta = vector.unsqueeze(0)
        else:
            # [channels] -> [1, 1, channels]
            #
            # PyTorch broadcasts this across batch and text-token
            # dimensions without creating a repeated copy.
            steering_delta = vector.view(1, 1, -1)

        steered = activation + scale * steering_delta

        if self.validate_runtime and not torch.isfinite(steered).all():
            raise RuntimeError("Steered activation contains " "NaN or Inf.")

        self.modified_calls += 1

        self.calls_by_location[location] = (
            self.calls_by_location.get(
                location,
                0,
            )
            + 1
        )

        relative_scale = self._estimate_relative_scale(
            activation=activation,
            vector=vector,
            effective_strength=(effective_strength),
        )

        self.relative_scale_by_location[location] = relative_scale

        self.effective_strength_by_location[location] = effective_strength

        source_block, source_step = source_location

        self.vector_source_by_location[location] = {
            "source_block": source_block,
            "source_step": ("*" if source_step == self.vector_store.WILDCARD_STEP else source_step),
            "path": self.vector_store.path_for(source_location),
        }

        return steered

    @staticmethod
    def _estimate_relative_scale(
        activation: torch.Tensor,
        vector: torch.Tensor,
        effective_strength: float,
    ) -> float:
        activation_norm = activation.detach().float().norm(dim=-1).mean().item()

        vector_norm = vector.detach().float().norm(dim=-1).mean().item()

        delta_norm = abs(effective_strength) * vector_norm

        return float(delta_norm / max(activation_norm, 1.0e-8))

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
            dict[str, Any],
        ] = {}

        self.vector_source_by_location: dict[
            str,
            dict[str, Any],
        ] = {}

    def clear_runtime_cache(self) -> None:
        self.vector_store.clear_runtime_cache()

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
            "mean": (sum(values) / len(values)),
            "max": max(values),
        }

    def statistics(self) -> dict[str, Any]:
        relative_scales = list(self.relative_scale_by_location.values())

        effective_strengths = list(self.effective_strength_by_location.values())

        probabilities = [record["p_cls"] for record in self.classifier_by_location.values()]

        eta_values = [record["eta_cls"] for record in self.classifier_by_location.values()]

        return {
            "vector_type": self.vector_type,
            "timing_mode": self.timing_mode,
            "vector_source_by_location": dict(self.vector_source_by_location),
            "operation": self.operation,
            "base_strength": self.strength,
            "use_classifier": (self.use_classifier),
            "total_calls": self.total_calls,
            "modified_calls": (self.modified_calls),
            "calls_by_location": dict(self.calls_by_location),
            "classifier_by_location": dict(self.classifier_by_location),
            "effective_strength_by_location": dict(self.effective_strength_by_location),
            "relative_scale_by_location": dict(self.relative_scale_by_location),
            "p_cls": self._summary_values(probabilities),
            "eta_cls": self._summary_values(eta_values),
            "effective_strength": (self._summary_values(effective_strengths)),
            "relative_scale": (self._summary_values(relative_scales)),
        }

    def summary(
        self,
    ) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "vector_type": self.vector_type,
            "timing_mode": self.timing_mode,
            "source_step": (self.source_step if self.timing_mode == "shared" else None),
            "available_blocks": sorted(self.available_blocks),
            "vector_store": (self.vector_store.summary()),
            "statistics": self.statistics(),
            "regularizer": (self.regularizer.summary() if self.regularizer is not None else None),
        }
