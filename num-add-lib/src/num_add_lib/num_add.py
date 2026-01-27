
import torch

from torch import Tensor
from torch.library import custom_op, register_kernel
from num_add_lib.cpp_extension_utils import register_extension

class SpecializedModule(torch.nn.Module):
 
    def __init__(self, number : int):
        super().__init__()
        self.number = number
        self.ns = f"test_number_{number}"
        self.lib = torch.library.Library(ns=self.ns, kind="DEF")

        def forward_op(x: Tensor, num : int) -> Tensor: 
            return x + num  
        
        self.forward_custom_op = custom_op( 
            self.ns + "::forward_op",
            forward_op,
            mutates_args=[],
            device_types=None,
            )
        
        # register_kernel(
        #     self.ns + "::forward_op",
        #     None,
        #     forward_op,
        #     lib = self.lib
        # )
        
        # register_extension(self.ns, self.number)
        

    def forward(self, x):
        return self.forward_custom_op(x, self.number)
        

    
