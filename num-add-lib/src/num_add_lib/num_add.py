from collections.abc import Callable
from enum import StrEnum
from typing import Any

import torch

from torch import Tensor
from torch.library import custom_op
from num_add_lib.cpp_extension_utils import register_cpp_extension


class CppRegistrationType(StrEnum):
    Forward = "CppForward"
    Nothing = "NoCpp"


def _as_tuple(argnums: int | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(argnums, int):
        return (argnums,)
    return tuple(argnums)


def make_autograd_registration(
    reference_forward: Callable[..., Any],
    *,
    diff_argnums: int | tuple[int, ...],
) -> tuple[Callable[..., None], Callable[..., tuple[Any, ...]]]:
    differentiable_argnums = _as_tuple(diff_argnums)

    def setup_context(ctx, inputs, output) -> None:
        del output
        ctx.diff_argnums = differentiable_argnums
        ctx.num_inputs = len(inputs)
        ctx.save_for_backward(*inputs)

    def backward(ctx, *grad_outputs):
        if all(grad_output is None for grad_output in grad_outputs):
            return (None,) * ctx.num_inputs

        if not any(ctx.needs_input_grad[idx] for idx in ctx.diff_argnums):
            return (None,) * ctx.num_inputs

        inputs = ctx.saved_tensors
        diff_inputs = tuple(inputs[idx] for idx in ctx.diff_argnums)

        def diff_only_forward(*updated_diff_inputs):
            full_inputs = list(inputs)
            for idx, updated_input in zip(ctx.diff_argnums, updated_diff_inputs):
                full_inputs[idx] = updated_input
            return reference_forward(*full_inputs)

        _, vjp_fn = torch.func.vjp(diff_only_forward, *diff_inputs)
        cotangents = grad_outputs[0] if len(grad_outputs) == 1 else grad_outputs
        diff_grads = vjp_fn(cotangents)

        full_grads = [None] * ctx.num_inputs
        for idx, grad in zip(ctx.diff_argnums, diff_grads):
            if ctx.needs_input_grad[idx]:
                full_grads[idx] = grad
        return tuple(full_grads)

    return setup_context, backward


class SpecializedModule(torch.nn.Module):
    def __init__(
        self,
        number: int,
        cpp_registration: CppRegistrationType,
    ):
        super().__init__()
        self.number_value = number
        self.number = torch.tensor(number, dtype=torch.int64)
        self.cpp_registration = cpp_registration
        self.ns = f"test_number_{number}_CustomOpAutograd_{cpp_registration}"

        def forward_op(x: Tensor, num: Tensor) -> Tensor:
            print("forward native pytorch")
            return x + num

        self.forward_custom_op = custom_op(
            self.ns + "::forward_op",
            forward_op,
            mutates_args=(),
            device_types=None,
        )

        @self.forward_custom_op.register_fake
        def _(x: Tensor, num: Tensor) -> Tensor:
            return forward_op(x, num)

        setup_context, backward = make_autograd_registration(
            forward_op,
            diff_argnums=(0,),
        )

        torch.library.register_autograd(
            self.forward_custom_op,
            backward,
            setup_context=setup_context,
        )

        match cpp_registration:
            case CppRegistrationType.Forward:
                register_cpp_extension(self.ns, self.number_value)
            case CppRegistrationType.Nothing:
                pass

    def forward(self, x):
        return self.forward_custom_op(x, self.number)
