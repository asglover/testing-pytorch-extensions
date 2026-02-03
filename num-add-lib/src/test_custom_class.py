import torch
import pytest

import logging

from num_add_lib import SpecializedModule, PythonRegistrationType, CppRegistrationType
from torch.utils._debug_mode import DebugMode

logger = logging.getLogger(__name__)


@pytest.fixture(params=[1], scope="module")
def num(request):
    return request.param

@pytest.fixture(params=list(PythonRegistrationType), scope="module")
def python_registration(request):
    return request.param

@pytest.fixture(params=list(CppRegistrationType), scope="module")
def cpp_registration(request):
    return request.param

@pytest.fixture(scope="module")
def module(num, python_registration, cpp_registration):
    return SpecializedModule(num, python_registration, cpp_registration)


@pytest.fixture(scope="function")
def t():
    return torch.Tensor([1, 4, 5, 6, 7])


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
