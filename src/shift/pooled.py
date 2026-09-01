from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.utils.hashing import sha256_file_set


class PooledSteeringController:
    """
    Apply steering to the FLUX pooled CLIP embedding.

    vector:
        mean(positive - negative)

    alpha_pool:
        gamma * cosine(initial, target)

    output:
        initial + sign * alpha_pool * vector
    """

    OPERATION_SIGNS: dict[str, float] = {
        "add": 1.0,
        "erase": -1.0,
    }

    VALID_SIMILARITY_MODES = {
        "raw",
        "positive",
        "absolute",
    }

    def __init__(
        self,
        vector_path: str,
        target_embedding_path: str,
        operation: str = "erase",
        strength: float = 0.0,
        enabled: bool = False,
        similarity_mode: str = "raw",
        normalize_vector: bool = False,
        eps: float = 1.0e-8,
    ) -> None:
        self.vector_path = Path(vector_path)

        self.target_embedding_path = Path(target_embedding_path)

        self.eps = float(eps)
        self.normalize_vector = bool(normalize_vector)

        if self.eps <= 0:
            raise ValueError("eps must be positive.")

        self._cpu_vector = self._load_tensor(self.vector_path)

        self._cpu_target_embedding = self._load_tensor(self.target_embedding_path)

        self.artifact_fingerprint = sha256_file_set(
            [
                ("vector", self.vector_path),
                (
                    "target_embedding",
                    self.target_embedding_path,
                ),
            ]
        )

        if self._cpu_vector.ndim != 1:
            raise RuntimeError(
                "Expected pooled vector shape "
                "[channels], got "
                f"{tuple(self._cpu_vector.shape)}."
            )

        if self._cpu_target_embedding.ndim != 1:
            raise RuntimeError(
                "Expected target embedding shape "
                "[channels], got "
                f"{tuple(self._cpu_target_embedding.shape)}."
            )

        if self._cpu_vector.shape != self._cpu_target_embedding.shape:
            raise RuntimeError("Pooled vector and target " "embedding shapes differ.")

        if self.normalize_vector:
            norm = self._cpu_vector.norm().clamp_min(self.eps)

            self._cpu_vector = self._cpu_vector / norm

        self._runtime_cache: dict[
            tuple[str, torch.dtype],
            tuple[
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}

        self.enabled = False
        self.operation = "erase"
        self.strength = 0.0
        self.similarity_mode = "raw"

        self.reset_statistics()

        self.configure(
            enabled=enabled,
            operation=operation,
            strength=strength,
            similarity_mode=(similarity_mode),
        )

    @staticmethod
    def _load_tensor(
        path: Path,
    ) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(path)

        try:
            value = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            value = torch.load(
                path,
                map_location="cpu",
            )

        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise TypeError(f"Expected tensor in {path}.")

        value = value.detach().float().cpu().contiguous()

        if not torch.isfinite(value).all():
            raise RuntimeError(f"Tensor in {path} contains " "NaN or Inf.")

        return value

    def configure(
        self,
        enabled: bool,
        operation: str,
        strength: float,
        similarity_mode: str | None = None,
    ) -> None:
        if operation not in self.OPERATION_SIGNS:
            raise ValueError(f"Unsupported pooled operation: " f"{operation!r}.")

        strength = float(strength)

        if strength < 0:
            raise ValueError("Pooled strength must be " "nonnegative.")

        next_similarity_mode = (
            self.similarity_mode if similarity_mode is None else str(similarity_mode)
        )

        if next_similarity_mode not in self.VALID_SIMILARITY_MODES:
            raise ValueError("Unknown similarity mode: " f"{next_similarity_mode!r}.")

        self.enabled = bool(enabled)
        self.operation = operation
        self.strength = strength
        self.similarity_mode = next_similarity_mode

    def apply(
        self,
        pooled_prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        self.total_calls += 1

        if not self.enabled or self.strength == 0.0:
            return pooled_prompt_embeds

        if pooled_prompt_embeds.ndim != 2:
            raise RuntimeError(
                "Expected pooled prompt embeddings "
                "[batch, channels], got "
                f"{tuple(pooled_prompt_embeds.shape)}."
            )

        (
            vector,
            target_embedding,
        ) = self._get_runtime_tensors(pooled_prompt_embeds)

        channels = pooled_prompt_embeds.shape[-1]

        if vector.shape[0] != channels:
            raise RuntimeError(
                "Pooled vector dimension "
                "does not match current embedding: "
                f"{vector.shape[0]} != {channels}."
            )

        batch_size = pooled_prompt_embeds.shape[0]

        target_batch = target_embedding.unsqueeze(0).expand(batch_size, -1)

        similarity = F.cosine_similarity(
            pooled_prompt_embeds.float(),
            target_batch.float(),
            dim=-1,
            eps=self.eps,
        )

        scaled_similarity = self._transform_similarity(similarity)

        alpha = self.strength * scaled_similarity

        sign = self.OPERATION_SIGNS[self.operation]

        delta = (
            sign * alpha.to(dtype=(pooled_prompt_embeds.dtype)).unsqueeze(-1) * vector.unsqueeze(0)
        )

        steered = pooled_prompt_embeds + delta

        if not torch.isfinite(steered).all():
            raise RuntimeError("Steered pooled embedding " "contains NaN or Inf.")

        initial_norm = pooled_prompt_embeds.detach().float().norm(dim=-1)

        delta_norm = delta.detach().float().norm(dim=-1)

        relative_scale = delta_norm / initial_norm.clamp_min(self.eps)

        self.modified_calls += 1

        self.last_record = {
            "enabled": self.enabled,
            "operation": self.operation,
            "strength": self.strength,
            "similarity_mode": (self.similarity_mode),
            "cosine_similarity": (similarity.detach().cpu().tolist()),
            "scaled_similarity": (scaled_similarity.detach().cpu().tolist()),
            "effective_alpha": (alpha.detach().cpu().tolist()),
            "initial_norm": (initial_norm.cpu().tolist()),
            "delta_norm": (delta_norm.cpu().tolist()),
            "relative_scale": (relative_scale.cpu().tolist()),
        }

        return steered

    def _transform_similarity(
        self,
        similarity: torch.Tensor,
    ) -> torch.Tensor:
        if self.similarity_mode == "raw":
            # Exact formula described in SHIFT.
            return similarity

        if self.similarity_mode == "positive":
            # Optional defensive ablation:
            # prevent negative cosine from
            # reversing the operation.
            return similarity.clamp_min(0.0)

        if self.similarity_mode == "absolute":
            return similarity.abs()

        raise RuntimeError("Invalid similarity mode.")

    def _get_runtime_tensors(
        self,
        reference: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        key = (
            str(reference.device),
            reference.dtype,
        )

        cached = self._runtime_cache.get(key)

        if cached is not None:
            return cached

        vector = self._cpu_vector.to(
            device=reference.device,
            dtype=reference.dtype,
            non_blocking=True,
        )

        target = self._cpu_target_embedding.to(
            device=reference.device,
            dtype=reference.dtype,
            non_blocking=True,
        )

        self._runtime_cache[key] = (
            vector,
            target,
        )

        return vector, target

    def reset_statistics(self) -> None:
        self.total_calls = 0
        self.modified_calls = 0
        self.last_record: dict[str, Any] | None = None

    def statistics(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "modified_calls": (self.modified_calls),
            "last_record": (self.last_record),
        }

    def configuration(self) -> dict[str, Any]:
        """Return static settings that affect pooled steering."""
        return {
            "type": self.__class__.__name__,
            "vector_path": str(self.vector_path),
            "target_embedding_path": str(
                self.target_embedding_path
            ),
            "normalize_vector": self.normalize_vector,
            "eps": self.eps,
            "artifact_fingerprint": (
                self.artifact_fingerprint
            ),
        }

    def summary(self) -> dict[str, Any]:
        return {
            **self.configuration(),
            "vector_shape": list(self._cpu_vector.shape),
            "vector_norm": float(self._cpu_vector.norm().item()),
            "target_embedding_norm": float(self._cpu_target_embedding.norm().item()),
            "normalize_vector": (self.normalize_vector),
            "enabled": self.enabled,
            "operation": self.operation,
            "strength": self.strength,
            "similarity_mode": (self.similarity_mode),
            "statistics": (self.statistics()),
        }
