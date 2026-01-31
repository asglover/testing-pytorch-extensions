import torch
import pytest

from num_add_lib import SpecializedModule

@pytest.fixture(params=[1,2,3], scope='module')
def num(request):
    return request.param

@pytest.fixture(scope='module')
def module(num):
    return SpecializedModule(num)


def test_forward(module):
    with torch.no_grad():
        t = torch.Tensor([1, 4, 5, 6, 7])
        print(t)
        test = module(t)
        ref = t + module.number
        print(test)
        print(ref)
        torch.testing.assert_close(test, ref)

def test_backward(module):
    t = torch.Tensor([1, 4, 5, 6, 7])
    t_custom = t.clone().requires_grad_()
    t_ref = t.clone().requires_grad_()

    loss_custom = torch.sum(module(t_custom))
    loss_ref = torch.sum(t_ref + module.number)

    loss_custom.backward()
    loss_ref.backward()

    torch.testing.assert_close(t_custom.grad, t_ref.grad)

    
    
