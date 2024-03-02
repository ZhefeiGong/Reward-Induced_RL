#@time   : Mar, 2024 
#@func   : the implementation of baseline model
#@author : Zhefei Gong
#@notice : 

import torch.nn as nn


#############################################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
#############################################################################
class MODEL_ORACLE(nn.Module):
    def __init__(self):
        super(MODEL_ORACLE, self).__init__()
    def w_init(self):
        pass
    def forward(self, observation):
        return observation

#############################################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
#############################################################################

