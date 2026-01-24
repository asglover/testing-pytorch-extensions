import torch
import pytest

from num_add_lib import SpecializedModule

@pytest.fixture(params=[1,2,3], scope='module')
def num(request):
    return request.param

@pytest.fixture(scope='module')
def module(num):
    return SpecializedModule(num)

@pytest.fixture
def t():
    return torch.Tensor([1, 4, 5, 6, 7])

def test_forward(module, t):
    print(t)
    print(module(t))
    print(t + module.number)
    torch.testing.assert_close(module(t), (t + module.number))

def test_backward(module, t):
    loss = torch.sum(module(t))
    loss.backward()
    