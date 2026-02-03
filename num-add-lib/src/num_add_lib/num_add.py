from enum import StrEnum

import torch

from torch import Tensor
from torch.library import custom_op, register_kernel
from num_add_lib.cpp_extension_utils import register_cpp_extension


class PythonRegistrationType(StrEnum):
    CompositeImplicitAutograd = "CompositeImplicitAutograd"
    CompositeExplicitAutograd = "CompositeExplicitAutograd"
    CustomOp = "CustomOp"


class CppRegistrationType(StrEnum):
    Forward = "CppForward"
    Autograd = "CppAutograd"
    Nothing = "NoCpp"


class SpecializedModule(torch.nn.Module):
    def __init__(
        self,
        number: int,
        python_registration: PythonRegistrationType,
        cpp_registration: bool,
    ):
        super().__init__()
        self.number = number
        self.ns = f"test_number_{number}_{python_registration}_{cpp_registration}"
        self.lib = torch.library.Library(ns=self.ns, kind="DEF")

        def forward_op(x: Tensor, num: int) -> Tensor:
            return x + num

        match python_registration:
            case PythonRegistrationType.CompositeImplicitAutograd:
                self.lib.define("forward_op(Tensor x, int num) -> Tensor")
                self.lib.impl("forward_op", forward_op, "CompositeImplicitAutograd")
                self.forward_custom_op = torch._C._jit_get_operation(
                    self.ns + "::forward_op"
                )[0]
            case PythonRegistrationType.CompositeExplicitAutograd:
                self.lib.define("forward_op(Tensor x, int num) -> Tensor")
                self.lib.impl("forward_op", forward_op, "CompositeExplicitAutograd")
                self.forward_custom_op = torch._C._jit_get_operation(
                    self.ns + "::forward_op"
                )[0]
            case PythonRegistrationType.CustomOp:
                self.forward_custom_op = custom_op(
                    self.ns + "::forward_op",
                    forward_op,
                    mutates_args=[],
                    device_types=None,
                )

        match cpp_registration:
            case CppRegistrationType.Forward:
                register_cpp_extension(self.ns, self.number, False)
            case CppRegistrationType.Autograd:
                register_cpp_extension(self.ns, self.number, True)
            case CppRegistrationType.Nothing:
                pass

    def forward(self, x):
        return self.forward_custom_op(x, self.number)
