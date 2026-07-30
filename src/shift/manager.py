from __future__ import annotations

from typing import Any

import torch
from torch.utils.hooks import RemovableHandle

from src.shift.hooks import (
    ShiftTextAttentionHook,
    TransformerStepHook,
)
from src.shift.state import ShiftRuntimeState


class NativeInterventionManager:
    """
    Original FLUX without registered SHIFT hooks.
    """

    def __init__(self) -> None:
        self._installed = False
        self._num_double_blocks = 0

    def install(
        self,
        transformer: torch.nn.Module,
    ) -> None:
        blocks = getattr(
            transformer,
            "transformer_blocks",
            None,
        )

        if blocks is None:
            raise AttributeError(
                "The FLUX transformer has no "
                "'transformer_blocks' attribute."
            )

        self._num_double_blocks = len(blocks)
        self._installed = True

    def remove(self) -> None:
        self._installed = False

    def report(self) -> dict[str, Any]:
        return {
            "type": "native",
            "installed": self._installed,
            "registered_blocks": 0,
            "num_double_blocks": (
                self._num_double_blocks
            ),
            "state": None,
        }


class ShiftInterventionManager:
    """
    Unified manager for disabled, collect and steer modes.
    """

    VALID_MODES = {
        "disabled",
        "collect",
        "steer",
    }

    def __init__(
        self,
        mode: str = "disabled",
        blocks: list[int] | None = None,
        steps: list[int] | None = None,
        collector: Any | None = None,
        controller: Any | None = None,
        pooled_controller: Any | None = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported mode={mode!r}. "
                f"Available: {sorted(self.VALID_MODES)}"
            )

        if mode == "collect" and collector is None:
            raise ValueError(
                "mode='collect' requires a collector."
            )

        if mode == "steer" and controller is None:
            raise ValueError(
                "mode='steer' requires a controller."
            )

        self.collector = collector
        self.controller = controller

        self.state = ShiftRuntimeState(
            mode=mode,
            active_blocks=(
                set(blocks)
                if blocks is not None
                else None
            ),
            active_steps=(
                set(steps)
                if steps is not None
                else None
            ),
        )

        self._text_handles: list[
            RemovableHandle
        ] = []

        self._step_handle: (
            RemovableHandle | None
        ) = None

        self._hook_objects: list[
            ShiftTextAttentionHook
        ] = []

        self._installed = False
        self._num_double_blocks = 0

        self.pooled_controller = pooled_controller

    def configure_pooled_steering(
        self,
        enabled: bool,
        operation: str,
        strength: float,
        similarity_mode: str | None = None,
    ) -> None:
        if self.pooled_controller is None:
            if enabled:
                raise RuntimeError(
                    "Pooled steering was enabled, "
                    "but no pooled controller is "
                    "configured."
                )

            return

        self.pooled_controller.configure(
            enabled=enabled,
            operation=operation,
            strength=strength,
            similarity_mode=similarity_mode,
        )

    def apply_pooled_steering(
        self,
        pooled_prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooled_controller is None:
            return pooled_prompt_embeds

        return self.pooled_controller.apply(
            pooled_prompt_embeds
        )

    def reset_pooled_statistics(
        self,
    ) -> None:
        if self.pooled_controller is not None:
            self.pooled_controller.reset_statistics()

    def pooled_statistics(
        self,
    ) -> dict[str, Any] | None:
        if self.pooled_controller is None:
            return None

        return (
            self.pooled_controller
            .statistics()
        )

    def configure_locations(
        self,
        blocks: list[int] | None,
        steps: list[int] | None,
    ) -> None:
        if (
            self.controller is not None
            and blocks is not None
        ):
            requested_blocks = {
                int(value)
                for value in blocks
            }

            missing_blocks = (
                requested_blocks
                - self.controller.available_blocks
            )

            if missing_blocks:
                raise ValueError(
                    "No steering vectors loaded for "
                    f"blocks: {sorted(missing_blocks)}"
                )

        self.state.set_active_locations(
            blocks=blocks,
            steps=steps,
        )

    def reset_steering_statistics(
        self,
    ) -> None:
        if self.controller is not None:
            self.controller.reset_statistics()

    def steering_statistics(
        self,
    ) -> dict[str, Any] | None:
        if self.controller is None:
            return None

        return self.controller.statistics()

    def install(
        self,
        transformer: torch.nn.Module,
    ) -> None:
        if self._installed:
            raise RuntimeError(
                "SHIFT hooks are already installed."
            )

        blocks = getattr(
            transformer,
            "transformer_blocks",
            None,
        )

        if blocks is None:
            raise AttributeError(
                "The FLUX transformer has no "
                "'transformer_blocks' attribute."
            )

        self._num_double_blocks = len(blocks)
        self.state.reset_traces()

        step_hook = TransformerStepHook(
            self.state
        )

        self._step_handle = (
            transformer.register_forward_pre_hook(
                step_hook,
                with_kwargs=True,
            )
        )

        for block_index, block in enumerate(
            blocks
        ):
            attention = getattr(
                block,
                "attn",
                None,
            )

            if attention is None:
                raise AttributeError(
                    f"Double-stream block {block_index} "
                    "has no 'attn' module."
                )

            text_output_projection = getattr(
                attention,
                "to_add_out",
                None,
            )

            if text_output_projection is None:
                raise AttributeError(
                    f"Attention in block {block_index} "
                    "has no 'to_add_out'."
                )

            hook = ShiftTextAttentionHook(
                block_index=block_index,
                state=self.state,
                collector=self.collector,
                controller=self.controller,
            )

            handle = (
                text_output_projection
                .register_forward_pre_hook(hook)
            )

            self._hook_objects.append(hook)
            self._text_handles.append(handle)

        self._installed = True

    def remove(self) -> None:
        if self._step_handle is not None:
            self._step_handle.remove()
            self._step_handle = None

        for handle in self._text_handles:
            handle.remove()

        self._text_handles.clear()
        self._hook_objects.clear()
        self._installed = False

    def begin_prompt_run(
        self,
        pair_name: str,
        prompt_role: str,
    ) -> None:
        self.state.begin_prompt_run(
            pair_name=pair_name,
            prompt_role=prompt_role,
        )

    def begin_steering_run(
        self,
        run_name: str,
    ) -> None:
        self.state.begin_steering_run(
            run_name=run_name
        )

    def configure_steering(
        self,
        operation: str,
        strength: float,
        use_classifier: bool = False,
    ) -> None:
        if self.controller is None:
            raise RuntimeError(
                "No steering controller is configured."
            )

        self.controller.configure(
            operation=operation,
            strength=strength,
            use_classifier=use_classifier,
        )

    def save_collected_activations(
        self,
    ) -> str | None:
        if self.collector is None:
            return None

        return str(self.collector.save())

    def report(self) -> dict[str, Any]:
        collector_summary = (
            self.collector.summary()
            if self.collector is not None
            else None
        )

        controller_summary = (
            self.controller.summary()
            if self.controller is not None
            else None
        )

        return {
            "type": "shift",
            "installed": self._installed,
            "registered_blocks": len(
                self._text_handles
            ),
            "num_double_blocks": (
                self._num_double_blocks
            ),
            "state": self.state.to_dict(),
            "collector": collector_summary,
            "controller": controller_summary,
            "pooled_controller": (
                self.pooled_controller.summary()
                if self.pooled_controller is not None
                else None
            ),
        }
