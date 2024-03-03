#@time   : Mar, 2024 
#@func   : the implementation of baseline model
#@author : Zhefei Gong
#@notice : 

import torch
import torch.nn as nn
import numpy as np
from model import ENCODER

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
class MODEL_CNN(nn.Module):
    #@func : 
    def __init__(self,
                 input_resolution = 64,
                 input_channels = 1,
                 latent_channels = 16):
        super(MODEL_CNN, self).__init__()
        
        self.input_resolution = input_resolution
        self.input_channels = input_channels
        self.latent_channels = latent_channels

        # [N,1,64,64]
        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channels, self.latent_channels, kernel_size=3, stride=2, padding=1),  # Height & Weight / 2 <<-->> 64 / 2 = 32
            nn.ReLU(),
            nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=3, stride=2, padding=1), # Height & Weight / 2 <<-->> 32 / 2 = 16
            nn.ReLU(),
            nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=3, stride=2, padding=1), # Height & Weight / 2 <<-->> 16 / 2 = 8
            nn.ReLU(),
            # -->> [N,16,8]
            nn.Flatten() 
            # -->> [N,16*8*8] = [N,1024]
        )
        
        self.output_size = int(self.latent_channels * (self.input_resolution/2/2/2) * (self.input_resolution/2/2/2))

    #@func : 
    def forward(self, observation):
        # numpy <<-->> tensor
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32)
        observation = observation.view(-1, self.input_channels, self.input_resolution, self.input_resolution)  # [N, 1, 64, 64]
        # calculate
        x = self.cnn(observation) # [N, self.output_size]

        # print('[CNN-HERE]',x.shape) # [N, 1024]
        return x

    #@func : 
    def w_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Conv2d - fan_in
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias
            elif isinstance(module, nn.ConvTranspose2d):
                # ConvTranspose2d - fan_in
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias
            elif isinstance(module, nn.Linear):
                # Linear - fan_out
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias


#############################################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
#############################################################################
class MODEL_IMAGE_SCRATCH(nn.Module):
    def __init__(self,
                 input_resolution = 64,
                 input_channels = 1,
                 output_channels = 64):
        super(MODEL_IMAGE_SCRATCH, self).__init__()
        
        self.input_resolution = input_resolution
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.encoder = ENCODER(self.input_resolution, self.input_channels, self.output_channels)

        self.output_size = output_channels

    def forward(self, observation):
        
        # numpy <<-->> tensor
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32)
        observation = observation.view(-1, self.input_channels, self.input_resolution, self.input_resolution)  # [N, 1, 64, 64]

        return self.encoder(observation)
        
    #@func : initialize the encoder randomly
    def w_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Conv2d - fan_in
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias
            elif isinstance(module, nn.ConvTranspose2d):
                # ConvTranspose2d - fan_in
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias
            elif isinstance(module, nn.Linear):
                # Linear - fan_out
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias

#############################################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
#############################################################################
class MODEL_REWARD_PREDICTION(nn.Module):
    #@func : 
    def __init__(self, 
                 w_path, 
                 is_finetune = True,
                 input_resolution = 64,
                 input_channels = 1,
                 output_channels = 64,):
        super(MODEL_REWARD_PREDICTION, self).__init__()
        
        self.input_resolution = input_resolution
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.output_size = output_channels

        self.encoder = ENCODER(self.input_resolution, self.input_channels, self.output_channels)

        self.w_path = w_path
        self.is_finetune = is_finetune

    #@func : 
    def forward(self, observation):
        
        # numpy <<-->> tensor
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32)
        observation = observation.view(-1, self.input_channels, self.input_resolution, self.input_resolution)  # [N, 1, 64, 64]

        return self.encoder(observation)
        
    #@func : 
    def w_init(self):
        # init weights
        self.encoder.load_state_dict(torch.load(self.w_path))
        # freeze the weights
        if self.is_finetune==False:
            for param in self.encoder.parameters():
                param.requires_grad = False

#############################################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
#############################################################################
class MODEL_IMAGE_RECONSTRUCTION(nn.Module):
    #@func : 
    def __init__(self, 
                 w_path, 
                 is_finetune = True,
                 input_resolution = 64,
                 input_channels = 1,
                 output_channels = 64,):
        super(MODEL_IMAGE_RECONSTRUCTION, self).__init__()
        
        self.input_resolution = input_resolution
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.output_size = output_channels
        
        self.encoder = ENCODER(input_resolution = self.input_resolution, 
                               input_channels = self.input_channels, 
                               output_channels = self.output_channels)

        self.w_path = w_path
        self.is_finetune = is_finetune

    #@func : 
    def forward(self, observation):
        
        # numpy <<-->> tensor
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32)
        observation = observation.view(-1, self.input_channels, self.input_resolution, self.input_resolution)  # [N, 1, 64, 64]

        return self.encoder(observation)
        
    #@func : 
    def w_init(self):
        # init weights
        self.encoder.load_state_dict(torch.load(self.w_path))
        # freeze the weights
        if self.is_finetune==False:
            for param in self.encoder.parameters():
                param.requires_grad = False