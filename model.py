#@time   : Feb, 2024 
#@func   : the implementation of 'The Reward Prediction Model'
#@author : Zhefei Gong
#@notice : use Kaiming init due to the ReLU function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#############################################################################
##@time   : 
##@author : Zhefei Gong
##@func   : encode the trajectory images into the trajectory vectors
#############################################################################
class ENCODER(nn.Module):
    #
    def __init__(self, input_resolution = 64, input_channels = 1, output_channels=64):
        super(ENCODER,self).__init__() # initialize the father class

        self.layers = nn.ModuleList()
        self.num_layers = int(math.log2(input_resolution))
        current_channels = 4

        #### first layer <<-->> C[input_channels->4] R[/2] ####
        # [N, 4, H/2, W/2]
        self.layers.append(nn.Conv2d(input_channels, current_channels, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU())

        #### other layers <<-->> C[*2] R[/2] ####
        for _ in range(self.num_layers - 1):
            # [N, C*2, H/2, W/2]
            self.layers.append(nn.Conv2d(current_channels, current_channels * 2, kernel_size=3, stride=2, padding=1))
            self.layers.append(nn.ReLU())
            current_channels *= 2
        # 
        self.map_layer = nn.Linear(current_channels, output_channels) # [N, output_channels]
    
    #
    def forward(self, x):
        # x has shape [N, input_channels, input_resolution, input_resolution]]
        for layer in self.layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1) # [N,current_channels,1,1] -> [N,current_channels]
        x = self.map_layer(x) # [N,current_channels] -> [N, output_channels]
        return x
    
    #
    def w_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Conv2d - fan_in
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias
            elif isinstance(module, nn.Linear):
                # Linear - fan_out
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias

##################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
##################################################
class DECODER(nn.Module):

    # 
    def __init__(self, input_channels = 64, output_resolution = 64, output_channels=1):
        super(DECODER, self).__init__()

        current_channels = output_resolution * 2 # because first we multiple with 4
        self.num_layers = int(math.log2(output_resolution))
        self.output_channels = output_channels
        
        #### mapping layer ####
        self.layers = nn.ModuleList()
        self.map_layer = nn.Linear(input_channels, current_channels) # 64 -> 128

        #### other layers <<-->> C[/2] R[*2] ####
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.ConvTranspose2d(current_channels, current_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1))
            self.layers.append(nn.ReLU())
            current_channels = current_channels // 2

        #### last layer <<-->> C[4->output_channels] R[*2] ####
        self.layers.append(nn.ConvTranspose2d(current_channels, self.output_channels, kernel_size=3, stride=2, padding=1, output_padding=1))   
        
        # self.layers.append(nn.ReLU())     # if the datasst input size is [-1, 1], so we shouldn't use ReLU to restrict the range of data
                                            # if the datasst input size is [0, 1], we shoudl use ReLU to restrict the range of data

    #
    def w_init(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                # Conv2d - fan_in
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias
            elif isinstance(module, nn.Linear):
                # Linear - fan_out
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias

    #
    def forward(self, x):

        x = self.map_layer(x) # [T,input_channels] -> [T, current_channels]
        x = F.relu(x)                     # 
        x = x.unsqueeze(-1).unsqueeze(-1) # [T, current_channels, 1, 1]
        for layer in self.layers:
            x = layer(x)                    # [T, output_channels, output_resolution, output_resolution]

        return x

##################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
##################################################
class MLPs(nn.Module):
    #
    def __init__(self, input_size, output_size, hidden_units=32):
        super(MLPs, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_size)
        )
    #
    def forward(self, x):
        x = self.mlp(x)
        return x
    
    #
    def w_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Conv2d - fan_in
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias
            elif isinstance(module, nn.Linear):
                # Linear - fan_out
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias

##################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
##################################################
class LSTM(nn.Module):
    #
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
    
    #
    def forward(self, x):
        x, _ = self.lstm(x) # (N, L, H_in) -> (N, L, D*H_in) & [D=2 if bidirectional=True otherwise 1]
        return x

##################################################
##@time   : 
##@author : Zhefei Gong
##@func   : Reward Prediction Model
##################################################
class MODEL_REWARD_PRD(nn.Module):
    #
    def __init__(self,
                input_resolution = 64, 
                input_channels = 1, 
                hidden_dim=64,
                mlp_hidden_units = 32,
                lstm_num_layers = 1,
                rewards = None,
                ):
        super().__init__()
        
        self.input_resolution = input_resolution
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_rewards = len(rewards) if rewards is not None else 0
        self.mlp_hidden_units = mlp_hidden_units
        self.lstm_num_layers = lstm_num_layers

        # encoder
        self.encoder = ENCODER(
                        input_resolution=self.input_resolution, 
                        input_channels = self.input_channels, 
                        output_channels = self.hidden_dim)
        # mlp
        self.mlp = MLPs(input_size=self.hidden_dim, 
                        output_size=self.hidden_dim,
                        hidden_units=self.mlp_hidden_units)
        # lstm
        self.lstm = LSTM(input_size=self.hidden_dim, 
                         hidden_size=self.hidden_dim,
                         num_layers=self.lstm_num_layers)
        # reward head mlp
        self.mlp_reward_heads = nn.ModuleList()
        for _ in range(self.num_rewards):
            self.mlp_reward_heads.append(MLPs(input_size=self.hidden_dim, output_size=1, hidden_units=mlp_hidden_units))
    
    #
    def forward(self, x):
        
        x = self.encoder(x)         # [T,1,H,W] -> [T,64]
        x = self.mlp(x)             # [T,64] -> [T,64]
        x = self.lstm(x)            # [T,64] -> [T,64]
        
        rewards = [] # size should be (num_reward_haed, T)
        for mlp_reward_head in self.mlp_reward_heads:
            reward = mlp_reward_head(x)     # [T, 64] -> reward: [T, 1]
            rewards.append(reward)          # 
        
        rewards = torch.stack(rewards, dim=0)   # [num_reward_haed, T, 1]
        rewards = torch.squeeze(rewards)        # cut down the dimensions which have one 1 value 
                                                # [num_reward_haed, T, 1] -> [num_reward_haed, T]
        
        return rewards

    #
    def w_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Conv2d - fan_in
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias
            elif isinstance(module, nn.ConvTranspose2d):
                # Conv2d - fan_in
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias
            elif isinstance(module, nn.Linear):
                # Linear - fan_out
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias

##################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
##################################################
class MODEL_IMAGE_REC(nn.Module):
    #
    def __init__(self,
                 resolution = 64,
                 image_channels = 1,
                 latent_size = 64):
        super(MODEL_IMAGE_REC, self).__init__()
        
        self.image_resolution = resolution
        self.image_channels = image_channels
        self.latent_size = latent_size

        self.encoder = ENCODER(input_resolution = self.image_resolution, 
                               input_channels = self.image_channels, 
                               output_channels = self.latent_size)
        
        self.decoder = DECODER(input_channels = self.latent_size, 
                               output_resolution = self.image_resolution, 
                               output_channels = self.image_channels)

    #
    def forward(self, images):
        x = self.encoder(images)
        images_rec = self.decoder(x)
        return images_rec

    #
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



if __name__ == "__main__":
    pass
