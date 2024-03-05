#@time   : Mar, 2024 
#@func   : the implementation of 'PPO' algorithm
#@author : Zhefei Gong
#@notice : 

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from general_utils import sum_independent_dims
from sprites_env.envs.sprites import SpritesEnv, SpritesStateEnv
from baseline import MODEL_ORACLE, MODEL_CNN, MODEL_IMAGE_SCRATCH, MODEL_REWARD_PREDICTION, MODEL_IMAGE_RECONSTRUCTION


"""
@intro : the pseudocode of Actor-Critic Style
@refer : cite from paper [ppo](https://arxiv.org/abs/1707.06347)
@refer : code refer from [PPO-Pytorch](https://github.com/nikhilbarhate99/PPO-PyTorch/tree/master)

Initialize policy network π(a|s, θ) and value network V(s, ϕ)
for each iteration do

    # ACTOR
    for each environment step do
        Sample action a_t ~ π(a|s_t, θ)
        Execute action a_t, observe reward r_t and new state s_{t+1}
    end for

    # CRITIC
    for t = 0 to T-1 do
        Compute discounted return R_t = Σ_{k=0}^{T-t} γ^k r_{t+k}
        Compute advantage A_t = R_t - V(s_t, ϕ)
    end for
    
    # UPDATE

    # Update policy network parameters θ to maximize policy gradient
    L^{PG}(θ) = Σ_t A_t * log π(a_t|s_t, θ)
    
    # Update value network parameters ϕ to minimize value loss
    L^{V}(ϕ) = Σ_t (R_t - V(s_t, ϕ))^2
    
    Optional: Add entropy term to the loss to encourage exploration
end for


"""

##########################################################################################################################################################
##@time     : 
##@author   : Zhefei Gong
##@func     : 
##@notice   : Continuous actions are usually considerd to be independent 
##@resource : [DiagGaussianDistribution](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/distributions.html#DiagGaussianDistribution)
##########################################################################################################################################################

class DiagGaussianDistribution:
    def __init__(self, action_dim):
        super(DiagGaussianDistribution, self).__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None
        self.distribution = None

    def proba_distribution_net(self, latent_dim, log_std_init):
        mean_actions = nn.Linear(latent_dim, self.action_dim) # Mean For Distribution [2,]
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=True) # Standard For Distribution [2,]
        return mean_actions, log_std
    
    def proba_distribution(self, mean_actions, log_std):
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std) # [N,2]
        return self

    def log_prob(self, actions) -> torch.tensor:
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> torch.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> torch.Tensor:
        return self.distribution.rsample()

##########################################################################################################################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
##########################################################################################################################################################
class MODEL_ACTOR_CRITIC(nn.Module):
    def __init__(self, 
                 observation_space, 
                 action_space,
                 spec):
        
        super(MODEL_ACTOR_CRITIC, self).__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space

        self.latent_size_net = spec.latent_size_net                         # --> 32
        self.output_size_policy = spec.output_size_policy                   # --> 64
        self.output_size_value = spec.output_size_value                     # --> 1

        self.action_dim = action_space.shape[0]                             # --> 2

        self.mode = spec.mode

        self.is_finetune = True

        self.reward_w_path = spec.reward_w_path
        self.reconstruction_w_path = spec.reconstruction_w_path
        
        self.input_resolution = spec.input_resolution
        self.input_channels = spec.input_channels
        self.cnn_latent_channels = spec.cnn_latent_channels
        self.output_channels = spec.output_channels

        # =================================
        # ============ ENCODER ============
        # =================================

        # cnn
        if self.mode == 'cnn':
            self.is_finetune = True
            self.encoder = MODEL_CNN(input_resolution = self.input_resolution,
                                     input_channels = self.input_channels,
                                     latent_channels = self.cnn_latent_channels)
        # scratch
        elif self.mode == 'image_scratch':
            self.is_finetune = True
            self.encoder = MODEL_IMAGE_SCRATCH(input_resolution = self.input_resolution,
                                               input_channels = self.input_channels,
                                               output_channels = self.output_channels)            
        # image
        elif self.mode == 'image_reconstruction':
            self.is_finetune = False
            # print('HERE-w/o')
            self.encoder = MODEL_IMAGE_RECONSTRUCTION(w_path = self.reconstruction_w_path, 
                                                      is_finetune = self.is_finetune,
                                                      input_resolution = self.input_resolution,
                                                      input_channels = self.input_channels,
                                                      output_channels = self.output_channels,)
        elif self.mode == 'image_reconstruction_finetune':
            self.is_finetune = True
            # print('HERE-w')
            self.encoder = MODEL_IMAGE_RECONSTRUCTION(w_path = self.reconstruction_w_path, 
                                                      is_finetune = self.is_finetune,
                                                      input_resolution = self.input_resolution,
                                                      input_channels = self.input_channels,
                                                      output_channels = self.output_channels,)
        # reward
        elif self.mode == 'reward_prediction':
            self.is_finetune = False
            # print('HERE-w/o')
            self.encoder = MODEL_REWARD_PREDICTION(w_path = self.reward_w_path, 
                                                   is_finetune = self.is_finetune,
                                                   input_resolution = self.input_resolution,
                                                   input_channels = self.input_channels,
                                                   output_channels = self.output_channels)
        elif self.mode == 'reward_prediction_finetune':
            self.is_finetune = True
            # print('HERE-w')
            self.encoder = MODEL_REWARD_PREDICTION(w_path = self.reward_w_path, 
                                                   is_finetune = self.is_finetune,
                                                   input_resolution = self.input_resolution,
                                                   input_channels = self.input_channels,
                                                   output_channels = self.output_channels)
        # oracle
        else:
            self.is_finetune = True
            self.encoder = MODEL_ORACLE(dim=observation_space.shape[0])

        # POLICY <<-->> OUTPUT{[N,32]} 
        self.policy_net = nn.Sequential(
            nn.Linear(self.encoder.output_size, self.latent_size_net),  # 64 <<-->> 32 / 64(CNN)
            nn.ReLU(),
            nn.Linear(self.latent_size_net, self.output_size_policy)    # 32 / 64(CNN) <<-->> [N,64]
        )

        # VLAUE <<-->> OUTPUT{[N,1]}
        self.value_net = nn.Sequential(
            nn.Linear(self.encoder.output_size, self.latent_size_net),  # 64 <<-->> 32 / 64(CNN)
            nn.ReLU(),
            nn.Linear(self.latent_size_net, self.output_size_value)     # 32 / 64(CNN) <<-->> [N,1]
        )
        
        # Distribution for Action <<--->> Initialization
        self.action_distribution = DiagGaussianDistribution(self.action_dim)

        # nn.Linear / nn.parameter <<--->> Initialization
        self.action_net, self.log_std = self.action_distribution.proba_distribution_net(latent_dim=self.output_size_policy, log_std_init=0.0)

    #@func : 
    def w_init(self):

        # encoder
        self.encoder.w_init()

        # policy_net
        for module in self.policy_net:
            if isinstance(module, nn.Linear):
                # Linear - fan_out
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias
        
        # value_net
        for module in self.value_net:
            if isinstance(module, nn.Linear):
                # Linear - fan_out
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # bias
        
    #@func : 
    def forward(self):
        raise NotImplementedError
    
    #@func : 
    def actor(self, observation):

        # if isinstance(observation, np.ndarray):
        #     obs = torch.tensor(observation, dtype=torch.float)
        
        # observation - [N,1,64,64]

        # calculate
        latent_representation = self.encoder(observation)    

        # Freeze or Not
        if self.is_finetune == False:
            # print("[INFO] FREEZE !!!")
            latent_representation = latent_representation.detach()
        
        policy_output = self.policy_net(latent_representation)
        value_output = self.value_net(latent_representation)
        
        # sample the next action
        action_mean = self.action_net(policy_output)
        distribution = self.action_distribution.proba_distribution(action_mean, self.log_std)

        action_sample = distribution.sample().view(-1, self.action_dim) # [N,2]
        action_sample_log_prob = distribution.log_prob(action_sample).view(-1,1)  # [N, 1]
        
        return action_sample.detach(), action_sample_log_prob.detach(), value_output.detach() # DETACH
    
    #@func : 
    def critic(self, observation, action):

        # if isinstance(observation, np.ndarray):
        #     obs = torch.tensor(observation, dtype=torch.float)
        
        # observation - [N,64,64]
        # action - [N,2]
        
        # calculate
        latent_representation = self.encoder(observation)
        
        # Freeze or Not
        if self.is_finetune == False:
            # print("[INFO] FREEZE !!!")
            latent_representation = latent_representation.detach()
        
        policy_output = self.policy_net(latent_representation)
        value_output = self.value_net(latent_representation)

        action_mean = self.action_net(policy_output)
        distribution = self.action_distribution.proba_distribution(action_mean, self.log_std)

        action_sample = action.view(-1, self.action_dim) # [N,2]
        action_sample_log_prob = distribution.log_prob(action_sample).view(-1, 1) # [N, 1]

        return value_output, action_sample_log_prob, distribution.entropy()


##########################################################################################################################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
##########################################################################################################################################################

class ROLLOUT_BUFFER:
    def __init__(self):
        
        self.actions = []       # [N,2]
        self.observations = []  # [N,1,64,64]
        self.log_probs = []     # [N,2]
        self.RTGs = []          # [N,]
        self.At = []            # [N,]

        self.rewards = []       # [N,[list]] - CPU
    
    def clear(self):

        del self.actions       # [N,2]
        del self.observations  # [N,64,64]
        del self.log_probs     # [N,2]
        del self.RTGs          # [N,]
        del self.At            # [N,]

        del self.rewards       # [N,[list]] - CPU

        self.__init__()
        

"""
@intro : the pseudocode of PPO, Actor-Critic Style
@refer : cite from paper [ppo](https://arxiv.org/abs/1707.06347)
@refer : code refer from [PPO-Pytorch](https://github.com/nikhilbarhate99/PPO-PyTorch/tree/master)
@refer : code refer from [cleanrl](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)

for iteration=1,2,... do
    
    for actor=1,2,...,N do
        Run policy \pi_{\theta_{old}} in environment for T timesteps
        Compute advantage estimates \hat{A}_1,...,\hat{A}_T
    end for

    Optimize surrogate L wrt \theta, with K epochs and minibatch M \leq NT
    \theta_{olf} <- \theta

end for

"""

##########################################################################################################################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
##########################################################################################################################################################
class MODEL_PPO(nn.Module):
    
    #@func : 
    def __init__(self, 
                 env,
                 device,
                 spec
                 ):
        super(MODEL_PPO, self).__init__()
        
        # COUNT
        self.count_timestep_total = 0
        self.count_timestep_per_batch = 0

        # INFO
        self.num_timestep_total = spec.num_timestep_total
        self.num_timestep_per_batch = spec.num_timestep_per_batch
        self.num_timestep_per_episode = spec.num_timestep_per_episode
        self.input_channels = spec.input_channels
        self.normalization_bias = spec.normalization_bias
        self.mode = spec.mode

        self.is_oracle = (self.mode == 'oracle')

        # PARA
        self.r_gamma = spec.r_gamma
        self.clip_epsilon = spec.clip_epsilon
        self.coef_value_loss = spec.coef_value_loss
        self.coef_entropy_loss = spec.coef_entropy_loss

        # ENV PARA
        self.env = env
        self.num_distractors = env.n_distractors

        self.observation_space = env.observation_space        
        self.input_resolution = self.observation_space.shape[0]

        self.action_space = env.action_space
        self.action_dim = self.action_space.shape[0]

        # INIT 
        self.agent = MODEL_ACTOR_CRITIC(observation_space=self.observation_space, 
                                        action_space=self.action_space,
                                        spec=spec) # in GPUs

        self.buffer = ROLLOUT_BUFFER() # part in GPUs

        self.device = device # the devide number
    
    #@func : 
    def forward(self):
        pass
    
    #@func : 
    def w_init(self): 
        self.agent.w_init() # only need to initialize the agent
    
    #@func : 
    def buffer_clear(self):
        self.buffer.clear()
    
    #@func : 
    def compute_RTGs(self):
        with torch.no_grad():
            # Calculate
            for episode_rewards in reversed(self.buffer.rewards):
                discounted_reward = 0
                for reward in reversed(episode_rewards):
                    discounted_reward = self.r_gamma * discounted_reward + reward
                    self.buffer.RTGs.insert(0, discounted_reward)
            # RTGs [Normalizing]
            self.buffer.RTGs = torch.tensor(self.buffer.RTGs, dtype=torch.float32)
            self.buffer.RTGs = self.buffer.RTGs.view(self.count_timestep_per_batch, -1) #[N,]
            self.buffer.RTGs = (self.buffer.RTGs - self.buffer.RTGs.mean()) / (self.buffer.RTGs.std() + self.normalization_bias)
            self.buffer.RTGs = self.buffer.RTGs.to(self.device) # to GPU

        # print("[RTGs]",self.buffer.RTGs.shape) # [N,1]

    #@func : 
    def compute_advantage_estimate(self):
        with torch.no_grad():
            values, _, _ = self.agent.critic(self.buffer.observations, self.buffer.actions)
            values = values.view(-1, 1) # [N, 1]
            self.buffer.At = self.buffer.RTGs - values
            self.buffer.At = (self.buffer.At - self.buffer.At.mean()) / (self.buffer.At.std() + self.normalization_bias)
            self.buffer.At = self.buffer.At.to(self.device) # to GPU

        # print("[AT]",self.buffer.At.shape) # [N,1]

    #@func : 
    def rollout(self):
        
        # init
        self.count_timestep_per_batch = 0

        # batch
        while self.count_timestep_per_batch < self.num_timestep_per_batch : 

            observation = self.env.reset()
            episode_rewards = []
            count_timestep_per_episode = 0

            # episode
            while count_timestep_per_episode < self.num_timestep_per_episode : 

                count_timestep_per_episode += 1
                self.count_timestep_per_batch += 1

                # ACTOR
                with torch.no_grad():
                    self.buffer.observations.append(observation)                    # [64,64] / [4,]
                    # to device 
                    if isinstance(observation, np.ndarray):
                        observation = torch.tensor(observation, dtype=torch.float).to(self.device)
                    # calculate
                    action, action_log_prob, _ = self.agent.actor(observation)      # [N,2], [N,1], _
                
                # environment transform
                action, action_log_prob = action.cpu(), action_log_prob.cpu()
                observation, reward, done, _ = self.env.step(action) # [64,64] / [4,] , [1,], [1,]

                # EXPEND episode_rewards
                episode_rewards.append(reward)

                # EXPEND buffer 
                self.buffer.actions.append(action)
                self.buffer.log_probs.append(action_log_prob)

                #
                if done : 
                    break
            
            # 
            self.buffer.rewards.append(episode_rewards) # [N,[num_timestep_per_episode]]

        # == Observations ==
        #self.buffer.observations = torch.tensor(self.buffer.observations, dtype=torch.float32)
        self.buffer.observations = torch.tensor(np.array(self.buffer.observations), dtype=torch.float32).detach()
        if self.is_oracle:
            self.buffer.observations = self.buffer.observations.view(self.count_timestep_per_batch,self.input_resolution) # [N,D]
        else:
            self.buffer.observations = self.buffer.observations.view(self.count_timestep_per_batch,self.input_channels,self.input_resolution,self.input_resolution) # [N,C,H,W]
        
        # == Actions ==
        #self.buffer.actions = torch.tensor(torch.stack(self.buffer.actions,dim=0), dtype=torch.float32)
        self.buffer.actions = torch.stack(self.buffer.actions,dim=0).clone().detach().type(dtype=torch.float32)
        self.buffer.actions = self.buffer.actions.view(self.count_timestep_per_batch, self.action_dim) # [N, 2]

        # == Log_Probs ==
        #self.buffer.log_probs = torch.tensor(torch.stack(self.buffer.log_probs,dim=0), dtype=torch.float32)
        self.buffer.log_probs = torch.stack(self.buffer.log_probs,dim=0).clone().detach().type(dtype=torch.float32)
        self.buffer.log_probs = self.buffer.log_probs.view(self.count_timestep_per_batch, 1) # [N, 1]

        # TO GPU
        self.buffer.observations = self.buffer.observations.to(self.device)
        self.buffer.actions = self.buffer.actions.to(self.device)
        self.buffer.log_probs = self.buffer.log_probs.to(self.device)

        # print(self.buffer.observations.shape) # [N, -] / [N,1,64,64]
        # print(self.buffer.actions.shape) # [N,2] 
        # print(self.buffer.log_probs.shape) # [N,1]


    #@func : 
    def update(self):

        values, actions_log_probs, dist_entropy = self.agent.critic(self.buffer.observations, self.buffer.actions)        
        values = values.view(-1,1) # [N,1]
        
        # if torch.equal(actions_log_probs, self.buffer.log_probs):
        #     print('EQUAL')
        
        ratios = torch.exp(actions_log_probs - self.buffer.log_probs) # [N,1] - [N,1] <<-->> \pi_{\theta} / \pi_{\theta_{old}}
        ratios = ratios.view(-1,1) # [N,1]
        
        surr1 = ratios * self.buffer.At # [N,1] * [N,1]
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * self.buffer.At # [N,1] * [N,1]
        
        policy_loss = - torch.min(surr1, surr2).mean()  # [N,1] <-> num <<-->> MAX
        value_loss = F.mse_loss(self.buffer.RTGs, values) # [N,1] <-> num <<-->> MIN
        entropy_loss = -torch.mean(dist_entropy) # [N,1] <-> num <<-->> MAX(exploration)
        
        # LOSS = VALUE_LOSS + POLICY_LOSS + ENTROPY_LOSS
        loss = self.coef_value_loss * value_loss + policy_loss + self.coef_entropy_loss * entropy_loss
        
        return loss
    

if __name__ == "__main__":
    pass