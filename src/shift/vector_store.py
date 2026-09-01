from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from src.utils.hashing import sha256_file_set

VectorLocation = tuple[int, int]


class SteeringVectorStore:
    """
    Loads steering vectors and maps runtime locations to vector files.

    shared:
        Every runtime step in block b uses vector
        (b, source_step).

    per_step:
        Runtime location (b, t) uses vector (b, t).

    custom:
        vector_paths explicitly maps runtime block/step
        locations to arbitrary vector files.

        A direct path at block level acts as a wildcard for
        every runtime step of that block.
    """

    VALID_TIMING_MODES = {
        "shared",
        "per_step",
        "custom",
    }

    WILDCARD_STEP = -1

    VECTOR_FILENAMES = {
        "tokenwise_difference": "step_{step:02d}_vector.pt",
        "token_mean_difference": ("step_{step:02d}_token_mean_vector.pt"),
        "svm_normal": "step_{step:02d}_svm_normal.pt",
    }

    VECTOR_NDIMS = {
        "tokenwise_difference": {2},
        "token_mean_difference": {1},
        "svm_normal": {1},
        # Custom mode may mix channel and token-wise vectors.
        "auto": {1, 2},
    }

    def __init__(
        self,
        vector_type: str,
        timing_mode: str = "shared",
        vector_paths: Mapping[Any, Any] | None = None,
        vector_directory: str | None = None,
        svm_normal_directory: str | None = None,
        block_indices: Sequence[int] | None = None,
        step_indices: Sequence[int] | None = None,
        source_step: int = 0,
    ) -> None:
        self.vector_type = self._validate_vector_type(vector_type)

        self.timing_mode = self._validate_timing_mode(timing_mode)

        self.source_step = int(source_step)

        resolved_paths = self._resolve_paths(
            vector_paths=vector_paths,
            vector_directory=vector_directory,
            svm_normal_directory=svm_normal_directory,
            block_indices=block_indices,
            step_indices=step_indices,
        )

        self.artifact_fingerprint = sha256_file_set(
            (
                f"block={block},step={step}",
                path,
            )
            for (
                block,
                step,
            ), path in resolved_paths.items()
        )

        self._cpu_vectors: dict[
            VectorLocation,
            torch.Tensor,
        ] = {}

        self._paths: dict[
            VectorLocation,
            str,
        ] = {}

        for location, path in sorted(resolved_paths.items()):
            self._cpu_vectors[location] = self._load_vector(
                location=location,
                path=path,
            )

            self._paths[location] = str(path)

        self._runtime_cache: dict[
            tuple[
                VectorLocation,
                str,
                torch.dtype,
            ],
            torch.Tensor,
        ] = {}

    @classmethod
    def _validate_vector_type(
        cls,
        value: str,
    ) -> str:
        value = str(value).strip()

        if value not in cls.VECTOR_NDIMS:
            raise ValueError(
                f"Unsupported vector_type={value!r}. " f"Available: {sorted(cls.VECTOR_NDIMS)}"
            )

        return value

    @classmethod
    def _validate_timing_mode(
        cls,
        value: str,
    ) -> str:
        value = str(value).strip()

        if value not in cls.VALID_TIMING_MODES:
            raise ValueError(
                f"Unsupported timing_mode={value!r}. "
                f"Available: "
                f"{sorted(cls.VALID_TIMING_MODES)}"
            )

        return value

    def _resolve_paths(
        self,
        vector_paths: Mapping[Any, Any] | None,
        vector_directory: str | None,
        svm_normal_directory: str | None,
        block_indices: Sequence[int] | None,
        step_indices: Sequence[int] | None,
    ) -> dict[VectorLocation, Path]:
        if self.timing_mode == "custom":
            return self._parse_custom_paths(vector_paths)

        if vector_paths:
            raise ValueError("vector_paths is only valid when " "timing_mode='custom'.")

        if self.vector_type == "auto":
            raise ValueError("vector_type='auto' is only valid when " "timing_mode='custom'.")

        if not block_indices:
            raise ValueError("block_indices is required.")

        blocks = [int(value) for value in block_indices]

        if self.timing_mode == "shared":
            source_steps = [self.source_step]

        else:
            if not step_indices:
                raise ValueError("step_indices is required when " "timing_mode='per_step'.")

            source_steps = [int(value) for value in step_indices]

        if self.vector_type == "svm_normal":
            if svm_normal_directory is None:
                raise ValueError("svm_normal_directory is required " "for SVM-normal vectors.")

            root = Path(svm_normal_directory)

        else:
            if vector_directory is None:
                raise ValueError("vector_directory is required " "for difference vectors.")

            root = Path(vector_directory)

        filename_template = self.VECTOR_FILENAMES[self.vector_type]

        return {
            (block, step): (root / f"block_{block:02d}" / filename_template.format(step=step))
            for block in blocks
            for step in source_steps
        }

    def _parse_custom_paths(
        self,
        vector_paths: Mapping[Any, Any] | None,
    ) -> dict[VectorLocation, Path]:
        if not vector_paths:
            raise ValueError("timing_mode='custom' requires " "vector_paths.")

        result: dict[
            VectorLocation,
            Path,
        ] = {}

        for raw_block, block_value in vector_paths.items():
            block = int(raw_block)

            # Shorthand:
            #
            # vector_paths:
            #   0: path/to/vector.pt
            #
            # The vector is used at every runtime step
            # for block 0.
            if isinstance(
                block_value,
                (str, Path),
            ):
                result[
                    (
                        block,
                        self.WILDCARD_STEP,
                    )
                ] = Path(block_value)

                continue

            if not isinstance(
                block_value,
                Mapping,
            ):
                raise TypeError(
                    "Each custom block entry must be "
                    "a path or step-to-path mapping, "
                    f"got "
                    f"{type(block_value).__name__}."
                )

            for raw_step, raw_path in block_value.items():
                step_text = str(raw_step).strip().lower()

                if step_text in {
                    "*",
                    "default",
                }:
                    step = self.WILDCARD_STEP
                else:
                    step = int(raw_step)

                location = (block, step)

                if location in result:
                    raise ValueError("Duplicate custom vector " f"location: {location}.")

                result[location] = Path(str(raw_path))

        if not result:
            raise ValueError("No custom vector paths were " "configured.")

        return result

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
        location: VectorLocation,
        path: Path,
    ) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError("Vector for location " f"{location} does not exist: {path}")

        value = self._torch_load(path)

        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise TypeError(f"Expected Tensor in {path}.")

        allowed_ndims = self.VECTOR_NDIMS[self.vector_type]

        if value.ndim not in allowed_ndims:
            raise RuntimeError(
                f"Vector type {self.vector_type!r} "
                "allows tensor dimensions "
                f"{sorted(allowed_ndims)}, got "
                f"shape {tuple(value.shape)} in "
                f"{path}."
            )

        if not torch.isfinite(value).all():
            raise RuntimeError(f"Vector in {path} contains " "NaN or Inf.")

        return (
            value.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .contiguous()
        )

    def resolve_source_location(
        self,
        block_index: int,
        runtime_step: int,
    ) -> VectorLocation:
        block = int(block_index)
        step = int(runtime_step)

        if self.timing_mode == "shared":
            location = (
                block,
                self.source_step,
            )
        else:
            location = (
                block,
                step,
            )

        if location in self._cpu_vectors:
            return location

        if self.timing_mode == "custom":
            wildcard_location = (
                block,
                self.WILDCARD_STEP,
            )

            if wildcard_location in self._cpu_vectors:
                return wildcard_location

        raise KeyError(
            "No steering vector resolves runtime " f"location " f"(block={block}, step={step})."
        )

    def validate_locations(
        self,
        blocks: Sequence[int] | None,
        steps: Sequence[int] | None,
    ) -> None:
        if blocks is None:
            blocks = sorted(self.available_blocks)

        if steps is None:
            if self.timing_mode == "shared":
                return

            raise ValueError("steps must be explicit for " f"timing_mode=" f"{self.timing_mode!r}.")

        missing: list[tuple[int, int]] = []

        for block in blocks:
            for step in steps:
                try:
                    self.resolve_source_location(
                        block_index=int(block),
                        runtime_step=int(step),
                    )
                except KeyError:
                    missing.append(
                        (
                            int(block),
                            int(step),
                        )
                    )

        if missing:
            raise ValueError("No steering vectors resolve " "runtime locations: " f"{missing}")

    def get(
        self,
        block_index: int,
        runtime_step: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor,
        VectorLocation,
    ]:
        source_location = self.resolve_source_location(
            block_index=block_index,
            runtime_step=runtime_step,
        )

        cache_key = (
            source_location,
            str(device),
            dtype,
        )

        vector = self._runtime_cache.get(cache_key)

        if vector is None:
            vector = self._cpu_vectors[source_location].to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            )

            self._runtime_cache[cache_key] = vector

        return vector, source_location

    @property
    def available_blocks(
        self,
    ) -> set[int]:
        return {block for block, _ in self._cpu_vectors}

    def path_for(
        self,
        location: VectorLocation,
    ) -> str:
        return self._paths[location]

    def clear_runtime_cache(
        self,
    ) -> None:
        self._runtime_cache.clear()

    @staticmethod
    def _display_step(
        step: int,
    ) -> int | str:
        if step == SteeringVectorStore.WILDCARD_STEP:
            return "*"

        return step

    def configuration(
        self,
    ) -> dict[str, Any]:
        return {
            "vector_type": self.vector_type,
            "timing_mode": self.timing_mode,
            "source_step": (self.source_step if self.timing_mode == "shared" else None),
            "artifact_fingerprint": (
                self.artifact_fingerprint
            ),
            "paths": [
                {
                    "block": block,
                    "step": self._display_step(step),
                    "path": self._paths[(block, step)],
                }
                for block, step in sorted(self._paths)
            ],
        }

    def summary(
        self,
    ) -> dict[str, Any]:
        document = self.configuration()

        document["vectors"] = [
            {
                "block": block,
                "step": self._display_step(step),
                "path": self._paths[(block, step)],
                "shape": list(self._cpu_vectors[(block, step)].shape),
            }
            for block, step in sorted(self._cpu_vectors)
        ]

        return document
