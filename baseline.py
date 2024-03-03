#@time   : Mar, 2024 
#@func   : the implementation of baseline model
#@author : Zhefei Gong
#@notice : 

import torch
import torch.nn as nn
import numpy as np


#############################################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
#############################################################################
class MODEL_ORACLE(nn.Module):
    def __init__(self, dim):
        super(MODEL_ORACLE, self).__init__()

        self.input_size = dim
        self.output_size = dim
        
    def w_init(self):
        pass

    def forward(self, observation):
        # numpy <<-->> tensor
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32)
        # [N, output_dim]
        observation = observation.view(-1,self.output_size)
        return observation


#############################################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
#############################################################################

