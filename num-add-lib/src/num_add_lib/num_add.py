from collections.abc import Callable
from enum import StrEnum
from typing import Any

import torch

from torch import Tensor
from torch.library import custom_op
from num_add_lib.cpp_extension_utils import (
    register_cpp_backward_extension,
    register_cpp_forward_extension,
)


class CppRegistrationType(StrEnum):
    Forward = "CppForward"
    Nothing = "NoCpp"


class CppBackwardRegistrationType(StrEnum):
    BackwardX = "CppBackwardX"
    Nothing = "NoCppBackward"


def _as_tuple(argnums: int | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(argnums, int):
        return (argnums,)
    return tuple(argnums)


def make_vjp_autograd_registration(
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


def make_backward_x_reference(
    reference_forward: Callable[[Tensor, Tensor], Tensor],
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    def backward_x_op(grad_output: Tensor, x: Tensor, num: Tensor) -> Tensor:
        _, vjp_fn = torch.func.vjp(lambda x_: reference_forward(x_, num), x)
        return vjp_fn(grad_output)[0].clone()

    return backward_x_op


def make_forward_autograd_registration(
    backward_x_custom_op: Callable[[Tensor, Tensor, Tensor], Tensor],
) -> tuple[Callable[..., None], Callable[..., tuple[Tensor | None, None]]]:
    def setup_context(ctx, inputs, output) -> None:
        del output
        ctx.save_for_backward(*inputs)

    def backward(ctx, grad_output):
        x, num = ctx.saved_tensors
        if grad_output is None:
            return None, None
        if not ctx.needs_input_grad[0]:
            return None, None
        return backward_x_custom_op(grad_output, x, num), None

    return setup_context, backward


class SpecializedModule(torch.nn.Module):
    def __init__(
        self,
        number: int,
        cpp_registration: CppRegistrationType,
        cpp_backward_registration: CppBackwardRegistrationType = (
            CppBackwardRegistrationType.Nothing
        ),
    ):
        super().__init__()
        self.number_value = number
        self.number = torch.tensor(number, dtype=torch.int64)
        self.cpp_registration = cpp_registration
        self.cpp_backward_registration = cpp_backward_registration
        self.ns = (
            f"test_number_{number}_CustomOpAutograd_"
            f"{cpp_registration}_{cpp_backward_registration}"
        )

        def forward_op(x: Tensor, num: Tensor) -> Tensor:
            print("forward native pytorch")
            return x + num

        backward_x_op = make_backward_x_reference(forward_op)

        self.forward_custom_op = custom_op(
            self.ns + "::forward_op",
            forward_op,
            mutates_args=(),
            device_types=None,
        )

        @self.forward_custom_op.register_fake
        def _(x: Tensor, num: Tensor) -> Tensor:
            return forward_op(x, num)

        self.backward_x_custom_op = custom_op(
            self.ns + "::backward_x_op",
            backward_x_op,
            mutates_args=(),
            device_types=None,
        )

        @self.backward_x_custom_op.register_fake
        def _(grad_output: Tensor, x: Tensor, num: Tensor) -> Tensor:
            del grad_output, num
            return torch.empty_like(x)

        # backward_x_op needs its own autograd formula so forward_op supports
        # higher-order derivatives when its backward path is routed through this op.
        backward_x_setup_context, backward_x_backward = make_vjp_autograd_registration(
            backward_x_op,
            diff_argnums=(0, 1),
        )

        torch.library.register_autograd(
            self.backward_x_custom_op,
            backward_x_backward,
            setup_context=backward_x_setup_context,
        )

        setup_context, backward = make_forward_autograd_registration(
            self.backward_x_custom_op,
        )

        torch.library.register_autograd(
            self.forward_custom_op,
            backward,
            setup_context=setup_context,
        )

        match cpp_registration:
            case CppRegistrationType.Forward:
                register_cpp_forward_extension(self.ns, self.number_value)
            case CppRegistrationType.Nothing:
                pass

        match cpp_backward_registration:
            case CppBackwardRegistrationType.BackwardX:
                register_cpp_backward_extension(self.ns, self.number_value)
            case CppBackwardRegistrationType.Nothing:
                pass

    def forward(self, x):
        return self.forward_custom_op(x, self.number)
