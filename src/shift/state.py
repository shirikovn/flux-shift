from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class BlockTrace:
    calls: int = 0
    first_shape: tuple[int, ...] | None = None
    last_shape: tuple[int, ...] | None = None
    dtype: str | None = None
    device: str | None = None

    def record(
        self,
        activation: torch.Tensor,
    ) -> None:
        shape = tuple(activation.shape)

        if self.first_shape is None:
            self.first_shape = shape

        self.last_shape = shape
        self.dtype = str(activation.dtype)
        self.device = str(activation.device)
        self.calls += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "first_shape": (
                list(self.first_shape)
                if self.first_shape is not None
                else None
            ),
            "last_shape": (
                list(self.last_shape)
                if self.last_shape is not None
                else None
            ),
            "dtype": self.dtype,
            "device": self.device,
        }


@dataclass
class ShiftRuntimeState:
    mode: str = "disabled"

    active_blocks: set[int] | None = None
    active_steps: set[int] | None = None

    current_run_name: str | None = None

    # Used by collection mode.
    current_pair_name: str | None = None
    current_prompt_role: str | None = None

    current_step: int = -1
    current_timestep: float | None = None

    blocks: dict[int, BlockTrace] = field(
        default_factory=dict
    )

    def reset_traces(self) -> None:
        self.blocks.clear()

    def set_active_locations(
        self,
        blocks: list[int] | None,
        steps: list[int] | None,
    ) -> None:
        self.active_blocks = (
            set(int(value) for value in blocks)
            if blocks is not None
            else None
        )

        self.active_steps = (
            set(int(value) for value in steps)
            if steps is not None
            else None
        )

    def begin_prompt_run(
        self,
        pair_name: str,
        prompt_role: str,
    ) -> None:
        if prompt_role not in {
            "negative",
            "positive",
        }:
            raise ValueError(
                f"Unknown prompt role: {prompt_role!r}"
            )

        self.current_run_name = (
            f"{pair_name}__{prompt_role}"
        )
        self.current_pair_name = pair_name
        self.current_prompt_role = prompt_role

        self._reset_step()

    def begin_steering_run(
        self,
        run_name: str,
    ) -> None:
        self.current_run_name = run_name

        self.current_pair_name = None
        self.current_prompt_role = None

        self._reset_step()

    def _reset_step(self) -> None:
        self.current_step = -1
        self.current_timestep = None

    def advance_step(
        self,
        timestep: Any,
    ) -> None:
        self.current_step += 1
        self.current_timestep = self._to_float(
            timestep
        )

    def record(
        self,
        block_index: int,
        activation: torch.Tensor,
    ) -> None:
        trace = self.blocks.setdefault(
            block_index,
            BlockTrace(),
        )
        trace.record(activation)

    def should_capture(
        self,
        block_index: int,
    ) -> bool:
        if self.mode != "collect":
            return False

        if self.current_pair_name is None:
            return False

        if self.current_prompt_role is None:
            return False

        return self._location_is_active(
            block_index
        )

    def should_steer(
        self,
        block_index: int,
    ) -> bool:
        if self.mode != "steer":
            return False

        if self.current_run_name is None:
            return False

        return self._location_is_active(
            block_index
        )

    def _location_is_active(
        self,
        block_index: int,
    ) -> bool:
        if (
            self.active_blocks is not None
            and block_index not in self.active_blocks
        ):
            return False

        if (
            self.active_steps is not None
            and self.current_step
            not in self.active_steps
        ):
            return False

        return True

    @staticmethod
    def _to_float(
        value: Any,
    ) -> float | None:
        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None

            return float(
                value.flatten()[0].item()
            )

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @property
    def total_calls(self) -> int:
        return sum(
            trace.calls
            for trace in self.blocks.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "current_run_name": (
                self.current_run_name
            ),
            "current_pair_name": (
                self.current_pair_name
            ),
            "current_prompt_role": (
                self.current_prompt_role
            ),
            "current_step": self.current_step,
            "current_timestep": (
                self.current_timestep
            ),
            "active_blocks": (
                sorted(self.active_blocks)
                if self.active_blocks is not None
                else None
            ),
            "active_steps": (
                sorted(self.active_steps)
                if self.active_steps is not None
                else None
            ),
            "total_calls": self.total_calls,
            "blocks": {
                str(block_index): trace.to_dict()
                for block_index, trace in sorted(
                    self.blocks.items()
                )
            },
        }
