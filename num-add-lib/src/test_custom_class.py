import torch
import pytest

import logging

from num_add_lib import SpecializedModule, CppRegistrationType
from torch.utils._debug_mode import DebugMode

logger = logging.getLogger(__name__)


def _available_test_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.accelerator.is_available():
        accelerator = torch.device(
            torch.accelerator.current_accelerator(check_available=True)
        )
        if accelerator.type != "cpu":
            devices.append(torch.device(f"{accelerator.type}:0"))
    return devices


@pytest.fixture(params=[1], scope="module")
def num(request):
    return request.param


@pytest.fixture(params=list(CppRegistrationType), scope="module")
def cpp_registration(request):
    return request.param


@pytest.fixture(
    params=_available_test_devices(),
    ids=lambda device: str(device),
    scope="module",
)
def device(request):
    return request.param


@pytest.fixture(scope="module")
def module(num, cpp_registration, device):
    specialized_module = SpecializedModule(num, cpp_registration)
    specialized_module.number = specialized_module.number.to(device)
    return specialized_module


@pytest.fixture(scope="function")
def t(device):
    return torch.tensor([1, 4, 5, 6, 7], dtype=torch.float32, device=device)


def test_forward(module, t):
    t.detach()
    with torch.no_grad():
        with DebugMode(record_stack_trace=True) as dm:
            test = module(t)
        ref = t + module.number

    logger.debug(f"{test=}")
    logger.debug(f"{ref=}")
    logger.debug("\n" + dm.debug_string())
    torch.testing.assert_close(test, ref)


def test_forward_opcheck(module, t):
    op_overload = getattr(getattr(torch.ops, module.ns), "forward_op")
    torch.library.opcheck(op_overload, (t, module.number))


def test_backward(module, t):
    t_test = t.clone().requires_grad_()
    t_ref = t.clone().requires_grad_()

    with DebugMode(record_stack_trace=True) as dm:
        loss_test = torch.sum(module(t_test))
        loss_test.backward()

    loss_ref = torch.sum(t_ref + module.number)

    loss_ref.backward()
    logger.debug(f"{t_test.grad}")
    logger.debug(f"{t_ref.grad}")

    logger.debug(dm.debug_string())
    torch.testing.assert_close(t_test.grad, t_ref.grad)


def test_double_backward(module, t):
    t_test = t.clone().requires_grad_()
    t_ref = t.clone().requires_grad_()
    grad_output = torch.linspace(
        0.5,
        1.5,
        steps=t.numel(),
        dtype=t.dtype,
        device=t.device,
    ).reshape_as(t)

    with DebugMode(record_stack_trace=True) as dm:
        loss_test = torch.sum(module(t_test).pow(3))
        first_grad_test = torch.autograd.grad(loss_test, t_test, create_graph=True)[0]
        second_grad_test = torch.autograd.grad(
            torch.sum(first_grad_test * grad_output), t_test
        )[0]

    loss_ref = torch.sum((t_ref + module.number).pow(3))
    first_grad_ref = torch.autograd.grad(loss_ref, t_ref, create_graph=True)[0]
    second_grad_ref = torch.autograd.grad(
        torch.sum(first_grad_ref * grad_output), t_ref
    )[0]

    logger.debug(f"{first_grad_test=}")
    logger.debug(f"{first_grad_ref=}")
    logger.debug(f"{second_grad_test=}")
    logger.debug(f"{second_grad_ref=}")
    logger.debug(dm.debug_string())

    torch.testing.assert_close(first_grad_test, first_grad_ref)
    torch.testing.assert_close(second_grad_test, second_grad_ref)
