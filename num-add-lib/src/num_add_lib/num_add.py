from enum import StrEnum

import torch

from torch import Tensor
from torch.library import custom_op
from num_add_lib.cpp_extension_utils import register_cpp_extension


class CppRegistrationType(StrEnum):
    Forward = "CppForward"
    Nothing = "NoCpp"


class SpecializedModule(torch.nn.Module):
    def __init__(
        self,
        number: int,
        cpp_registration: CppRegistrationType,
    ):
        super().__init__()
        self.number = number
        self.cpp_registration = cpp_registration
        self.ns = f"test_number_{number}_CustomOpAutograd_{cpp_registration}"

        def forward_op(x: Tensor, num: int) -> Tensor:
            return x + num

        def scalarized_forward_reference(
            x: Tensor, num: int, grad_output: Tensor
        ) -> Tensor:
            return torch.sum(forward_op(x, num) * grad_output)

        self.forward_custom_op = custom_op(
            self.ns + "::forward_op",
            forward_op,
            mutates_args=(),
            device_types=None,
        )

        @self.forward_custom_op.register_fake
        def _(x: Tensor, num: int) -> Tensor:
            return forward_op(x, num)

        reference_backward_x = torch.func.grad(
            scalarized_forward_reference,
            argnums=0,
        )

        def setup_context(ctx, inputs, output) -> None:
            del output
            x, num = inputs
            ctx.num = num
            if ctx.needs_input_grad[0]:
                ctx.save_for_backward(x)

        def backward_without_grad_output(ctx, grad_output):
            del ctx, grad_output
            return None, None

        def backward_without_input_grad(ctx, grad_output):
            del ctx, grad_output
            return None, None

        def backward_with_input_grad(ctx, grad_output):
            (x,) = ctx.saved_tensors
            return reference_backward_x(x, ctx.num, grad_output), None

        def backward(ctx, grad_output):
            if grad_output is None:
                return backward_without_grad_output(ctx, grad_output)
            if not ctx.needs_input_grad[0]:
                return backward_without_input_grad(ctx, grad_output)
            return backward_with_input_grad(ctx, grad_output)

        torch.library.register_autograd(
            self.forward_custom_op,
            backward,
            setup_context=setup_context,
        )

        match cpp_registration:
            case CppRegistrationType.Forward:
                register_cpp_extension(self.ns, self.number)
            case CppRegistrationType.Nothing:
                pass

    def forward(self, x):
        return self.forward_custom_op(x, self.number)
