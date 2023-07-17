import numpy as np
import torch
import torch.nn as nn

class MaxCoord():
    def __init__(self):
        pass

    def update_output(self, input, sp_x, sp_y):
        input_dim = input.dim()
        assert input.dim() == 4, "Input must be 3D or 4D(batch)."
        assert input.size(0) == 1, "The first dimension of input has to be 1!"

        output = torch.zeros_like(input)

        v_max,c_max = torch.max(input, 1) 

        c_max_flatten = c_max.view(-1)
        v_max_flatten = v_max.view(-1)

        ind = c_max_flatten
        
        return output, ind,  v_max_flatten
