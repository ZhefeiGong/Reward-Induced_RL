#@time   : Mar, 2024 
#@func   : the implementation of 'PPO' algorithm
#@author : Zhefei Gong
#@notice : 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from general_utils import sum_independent_dims


from baseline import MODEL_ORACLE


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

#############################################################################
##@time     : 
##@author   : Zhefei Gong
##@func     : 
##@notice   : Continuous actions are usually considerd to be independent 
##@resource : [DiagGaussianDistribution](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/distributions.html#DiagGaussianDistribution)
#############################################################################

class DiagGaussianDistribution:
    def __init__(self, action_dim):
        super(DiagGaussianDistribution, self).__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None
        self.distribution = None

    def proba_distribution_net(self, latent_dim, log_std_init):
        mean_actions = nn.Linear(latent_dim, self.action_dim) # Mean For Distribution
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=True) # Standard For Distribution
        return mean_actions, log_std
    
    def proba_distribution(self, mean_actions, log_std):
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions) -> torch.tensor:
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> torch.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> torch.Tensor:
        return self.distribution.rsample()

#############################################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
#############################################################################
class MODEL_ACTOR_CRITIC(nn.Module):
    def __init__(self, 
                 observation_space, 
                 action_space,
                 mode = 'oracle', 
                 latent_size=64):
        super(MODEL_ACTOR_CRITIC, self).__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space

        self.input_size = observation_space.shape[0] # resolution -> 64
        self.latent_size = latent_size

        self.action_dim = action_space.shape[0] # 2
        self.value_dim = 1

        self.mode = mode

        # ENCODER
        if self.mode == 'oracle':
            self.encoder = MODEL_ORACLE()
        else:
            self.encoder = MODEL_ORACLE()

        # POLICY <<-->> OUTPUT
        self.policy_net = nn.Sequential(
            nn.Linear(self.input_size, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.latent_size)
        )

        # VLAUE <<-->> OUTPUT
        self.value_net = nn.Sequential(
            nn.Linear(self.input_size, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.value_dim)
        )

        # Distribution for Action <<--->> Initialization
        self.action_distribution = DiagGaussianDistribution(self.action_dim)

        # nn.Linear / nn.parameter <<--->> Initialization
        self.action_net, self.log_std = self.action_distribution.proba_distribution_net(latent_dim=self.latent_size, log_std_init=0.0)
    
    #@func : 
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
    
    #@func : 
    def forward(self):
        raise NotImplementedError
    
    #@func : 
    def actor(self, observation):

        # if isinstance(observation, np.ndarray):
        #     obs = torch.tensor(observation, dtype=torch.float)

        # observation - [N,64,64]
        
        # calculate
        latent_representation = self.encoder(observation)    
        policy_output = self.policy_net(latent_representation)
        value_output = self.value_net(latent_representation)

        # sample the next action
        action_mean = self.action_net(policy_output)
        distribution = self.action_distribution.proba_distribution(action_mean, self.log_std)

        action_sample = distribution.sample()
        action_sample_log_prob = distribution.log_prob(action_sample)
        
        return action_sample.detach(), action_sample_log_prob.detach(), value_output.detach() # DETACH
    
    #@func : 
    def critic(self, observation, action):

        # if isinstance(observation, np.ndarray):
        #     obs = torch.tensor(observation, dtype=torch.float)

        # observation - [N,64,64]
        # action - [N,2]

        # calculate
        latent_representation = self.encoder(observation)
        policy_output = self.policy_net(latent_representation)
        value_output = self.value_net(latent_representation)

        action_mean = self.action_net(policy_output)
        distribution = self.action_distribution.proba_distribution(action_mean, self.log_std)

        action_sample = action.view(-1, self.action_dim) # [N,2]
        action_sample_log_prob = distribution.log_prob(action_sample).view(-1, 1)

        return value_output, action_sample_log_prob, distribution.entropy()


#############################################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
#############################################################################

class ROLLOUT_BUFFER:
    def __init__(self):
        
        self.actions = []       # [N,2]
        self.observations = []  # [N,64,64]
        self.log_probs = []     # [N,2]
        self.RTGs = []          # [N,]
        self.At = []            # [N,]

        self.rewards = []       # [N,[list]] * 
    
    def clear(self):

        del self.actions[:]       # [N,2]
        del self.observations[:]  # [N,64,64]
        del self.log_probs[:]     # [N,2]
        del self.RTGs[:]          # [N,]
        del self.At[:]            # [N,]

        del self.rewards[:]       # [N,[list]] * 
        

"""
@intro : the pseudocode of PPO, Actor-Critic Style
@refer : cite from paper [ppo](https://arxiv.org/abs/1707.06347)
@refer : code refer from [PPO-Pytorch](https://github.com/nikhilbarhate99/PPO-PyTorch/tree/master)

for iteration=1,2,... do
    
    for actor=1,2,...,N do
        Run policy \pi_{\theta_{old}} in environment for T timesteps
        Compute advantage estimates \hat{A}_1,...,\hat{A}_T
    end for

    Optimize surrogate L wrt \theta, with K epochs and minibatch M \leq NT
    \theta_{olf} <- \theta

end for

"""

#############################################################################
##@time   : 
##@author : Zhefei Gong
##@func   : 
#############################################################################
class MODEL_PPO:
    
    #@func : 
    def __init__(self, 
                 env,
                 spec
                 ):
        
        # COUNT
        self.count_timestep_total = 0
        self.count_timestep_per_batch = 0

        # INFO
        self.num_timestep_total = spec.num_timestep_total
        self.num_timestep_per_batch = spec.num_timestep_per_batch
        self.num_timestep_per_episode = spec.num_timestep_per_episode
        self.input_channels = spec.input_channels
        self.normalization_bias = spec.normalization_bias
        self.latent_size = spec.latent_size
        self.mode = spec.mode

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
                                        mode = self.mode, 
                                        latent_size=self.latent_size
                                        )

        self.buffer = ROLLOUT_BUFFER()

    
    #@func : 
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

    #@func : 
    def buffer_clear(self):
        self.buffer.clear()
        
    #@func : 
    def compute_RTGs(self):

        # RTGs
        with torch.no_grad():
            # Calculate
            for episode_rewards in reversed(self.buffer.rewards):
                discounted_reward = 0
                for reward in reversed(episode_rewards):
                    discounted_reward = self.r_gamma * discounted_reward + reward
                    self.buffer.RTGs.insert(0, discounted_reward)

        # RTGs [Normalizing]
        self.buffer.RTGs = torch.tensor(self.buffer.RTGs, dtype=torch.float32)
        self.buffer.RTGs = self.buffer.RTGs.view(self.count_timestep_per_batch, -1)
        self.buffer.RTGs = (self.buffer.RTGs - self.buffer.RTGs.mean()) / (self.buffer.RTGs.std() + self.normalization_bias)

    #@func : 
    def compute_advantage_estimate(self):
        with torch.no_grad():
            values, _, _ = self.agent.critic(self.buffer.observations, self.buffer.actions)
            values = values.view(-1)
            self.buffer.At = self.buffer.RTGs - values
            self.buffer.At = (self.buffer.At - self.buffer.At.mean()) / (self.buffer.At.std() + self.normalization_bias)

    #@func : 
    def rollout(self):

        count_timestep_per_episode = 0
        self.count_timestep_per_batch = 0

        # batch
        while self.count_timestep_per_batch < self.num_timestep_per_batch : 

            observation = self.env.reset()
            episode_rewards = []

            # episode
            while count_timestep_per_episode < self.num_timestep_per_episode : 

                count_timestep_per_episode += 1
                self.count_timestep_per_batch += 1

                # ACTOR
                with torch.no_grad():
                    self.buffer.observations.append(observation)                     # [64,64]
                    action, action_log_prob, _ = self.agent.actor(observation)   # [2] / [2] / [1]
                
                # environment transform
                observation, reward, done, _ = self.env.step(action)

                # 
                episode_rewards.append(reward)

                # 
                self.buffer.actions.append(action)
                self.buffer.log_probs.append(action_log_prob)

                #
                if done : 
                    break
            
            # 
            self.buffer.rewards.append(episode_rewards) # [N,[num_timestep_per_episode]]

        # Observations
        self.buffer.observations = torch.tensor(self.buffer.observations, dtype=torch.float32)
        self.buffer.observations = self.buffer.observations.view(self.count_timestep_per_batch,self.input_channels,self.input_resolution,self.input_resolution)
        # Actions
        self.buffer.actions = torch.tensor(self.buffer.actions, dtype=torch.float32)
        self.buffer.actions = self.buffer.actions.view(self.count_timestep_per_batch, self.action_dim)
        # Log_probs
        self.buffer.log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32)
        self.buffer.log_probs = self.buffer.log_probs.view(self.count_timestep_per_batch, self.action_dim)

        # RTGs ...

    #@func : 
    def update(self):

        values, actions_log_probs, dist_entropy = self.agent.critic(self.buffer.observations, self.buffer.actions)
        
        values = values.view(-1)
        
        ratios = torch.exp(actions_log_probs - self.buffer.log_probs)
        ratios = ratios.view(-1)
        
        surr1 = ratios * self.buffer.At
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * self.buffer.At
        
        policy_loss = - torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(self.buffer.RTGs, values)
        entropy_loss = -torch.mean(dist_entropy)
        
        total_loss = self.coef_value_loss * value_loss + policy_loss + self.coef_entropy_loss * entropy_loss

        return total_loss
    

if __name__ == "__main__":
    pass