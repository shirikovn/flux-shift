from __future__ import annotations

from typing import Any

import torch

from src.shift.state import ShiftRuntimeState


class TransformerStepHook:
    """
    Called once at the beginning of every FLUX transformer pass.
    """

    def __init__(
        self,
        state: ShiftRuntimeState,
    ) -> None:
        self.state = state

    def __call__(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del module

        timestep = kwargs.get(
            "timestep",
            None,
        )

        # Retain the fallback that already worked
        # in the collection experiment.
        if timestep is None and len(args) >= 4:
            timestep = args[3]

        self.state.advance_step(timestep)
        return None


class ShiftTextBlockOutputHook:
    """
    Hook the output of a complete FLUX double-stream block.

    Diffusers returns:

        (encoder_hidden_states, hidden_states)
        (text tokens, image tokens)

    SHIFT is applied to the text residual representation that the
    next transformer block receives. This matches the activation
    location used by the official ControlGenAI/SHIFT FLUX launcher.

    Depending on state.mode it:
      - disabled: observes only;
      - collect: saves selected activations;
      - steer: replaces selected activations.
    """

    def __init__(
        self,
        block_index: int,
        state: ShiftRuntimeState,
        collector: Any | None = None,
        controller: Any | None = None,
    ) -> None:
        self.block_index = block_index
        self.state = state
        self.collector = collector
        self.controller = controller

    def __call__(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> tuple[Any, ...] | None:
        del module
        del inputs

        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError(
                "Expected a FLUX double-stream block output tuple "
                "(text_tokens, image_tokens), got "
                f"{type(output)!r}."
            )

        activation = output[0]

        if not isinstance(
            activation,
            torch.Tensor,
        ):
            raise TypeError(
                "Expected the first FLUX block output to be "
                "a text-token Tensor, got "
                f"{type(activation)!r}."
            )

        self.state.record(
            block_index=self.block_index,
            activation=activation,
        )

        if self.collector is not None and self.state.should_capture(self.block_index):
            self.collector.add(
                pair_name=str(self.state.current_pair_name),
                prompt_role=str(self.state.current_prompt_role),
                block_index=self.block_index,
                step_index=self.state.current_step,
                timestep=self.state.current_timestep,
                activation=activation,
            )

        if self.controller is not None and self.state.should_steer(self.block_index):
            steered = self.controller.apply(
                block_index=self.block_index,
                step_index=self.state.current_step,
                activation=activation,
            )

            # strength=0 returns the exact original tensor.
            if steered is activation:
                return None

            return (steered, *output[1:])

        return None
