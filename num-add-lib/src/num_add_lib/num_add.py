import random 
import torch

from torch import Tensor
from torch.library import custom_op

class SpecializedModule(torch.nn.Module):
 
    def __init__(self, number : int):
        super().__init__()
        self.number = number
        self.ns = f"test_number_{number}"
        self.lib = torch.library.Library(ns=self.ns, kind="DEF")

        
        def forward(x : Tensor)-> Tensor:
            return self.forward_custom_op(x, self.number)
        
        def forward_op(x: Tensor, num : int) -> Tensor: 
            return x + num 
        
        
        self.forward_custom_op = custom_op( 
            self.ns + "::forward_op",
            forward_op,
            mutates_args=[],
            device_types=None,
            )
        

        
    def forward(self, x):
        return self.forward_custom_op(x, self.number)
        
        

    
