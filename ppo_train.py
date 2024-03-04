#@time   : Mar, 2024 
#@func   : 
#@author : Zhefei Gong

import gym
import torch
import wandb
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import argparse
from general_utils import AttrDict
from general_utils import make_gif
from ppo import MODEL_PPO

#@func : 
def train_ppo(args):

    # WANDB
    if args.is_use_wandb:
        wandb.init(project=args.wandb_project, 
                name=args.wandb_exp,
                config=args)

    # PARA
    num_timestep_total = args.num_timestep_total
    num_timestep_per_batch = args.num_timestep_per_batch
    num_timestep_per_episode = args.num_timestep_per_episode
    num_update_per_batch = args.num_update_per_batch
    learning_rate_ppo = args.learning_rate_ppo
    max_grad_norm = args.max_grad_norm

    # INIT
    count_timestep_total = 0

    # ENV
    env_id = f"{'SpritesState' if args.mode == 'oracle' else 'Sprites'}-v{args.num_distractors}"
    env = gym.make(env_id)

    # MODEL
    _spec = AttrDict(
        
        # PPO
        mode = args.mode,
        num_timestep_total = args.num_timestep_total,
        num_timestep_per_batch = args.num_timestep_per_batch,
        num_timestep_per_episode = args.num_timestep_per_episode,
        normalization_bias = args.normalization_bias,
        r_gamma = args.r_gamma,
        clip_epsilon = args.clip_epsilon,
        coef_value_loss = args.coef_value_loss,
        coef_entropy_loss = args.coef_entropy_loss,

        # MODEL
        latent_size_net = args.latent_size_net, # 32/64
        input_channels = args.input_channels, # 1
        input_resolution = args.input_resolution, # 64
        cnn_latent_channels = args.cnn_latent_channels, # 16
        output_channels = args.output_channels, # 64

        output_size_policy = args.output_size_policy, # 64
        output_size_value = args.output_size_value, # 64

        reward_w_path = args.reward_w_path, # PATH
        reconstruction_w_path = args.reconstruction_w_path # PATH

    )

    ppo_agent = MODEL_PPO(env=env, spec=_spec)
    ppo_agent.w_init() # initialize the weights

    # OPTIMIZER
    optimizer = torch.optim.Adam(ppo_agent.agent.parameters(), lr=learning_rate_ppo)

    # TRAIN
    pbar = tqdm(total=int((num_timestep_total + num_timestep_per_batch - 1 )  / num_timestep_per_batch), desc="Training Progress")

    while count_timestep_total < num_timestep_total:
        
        # INIT
        avg_reward_per_batch = 0
        avg_loss_per_batch = 0
        
        # ACTOR
        with torch.no_grad():
            # RUN
            ppo_agent.rollout()
            ppo_agent.compute_RTGs()
            ppo_agent.compute_advantage_estimate()
            count_timestep_total += ppo_agent.count_timestep_per_batch   
            
            # SHOW the Figs([N,Resolution,Resolution])  
            if args.mode != 'oracle' and args.is_visual_traj is True:    
                imgs = torch.squeeze(ppo_agent.buffer.observations[:num_timestep_per_episode,:,:,:],dim=1) # [N,1,R,R] <<-->> [N,R,R]
                make_gif(imgs = np.array(imgs) * 255, path = "tmp/fig3/env.gif", fps_default=10)
                # print('[SAVING]', count_timestep_total)

        # CRITIC
        with torch.enable_grad():
            for _ in range(num_update_per_batch):
                optimizer.zero_grad()                   # <<-->> clear accumulated gradient
                loss = ppo_agent.update()               # <<-->> update
                loss.backward()                         # <<-->> backwards
                avg_loss_per_batch += loss.item()       # record
                nn.utils.clip_grad_norm_(ppo_agent.agent.parameters(), max_grad_norm) # <<-->> clip
                optimizer.step()                        # <<-->> optimize
        
        # tqdm
        sum_rewards = sum([item for sublist in ppo_agent.buffer.rewards for item in sublist])
        avg_reward_per_batch = sum_rewards / ppo_agent.count_timestep_per_batch * num_timestep_per_episode
        avg_loss_per_batch = avg_loss_per_batch / num_update_per_batch
        pbar.update(1)
        pbar.set_postfix({'reward': avg_reward_per_batch, 'loss': avg_loss_per_batch, 'save':(args.mode != 'oracle' and args.is_visual_traj is True)})
        
        # WANDB
        if args.is_use_wandb:
            wandb.log({"reward": avg_reward_per_batch}, step=count_timestep_total)
            wandb.log({'loss': avg_loss_per_batch}, step=count_timestep_total)

        # CLEAR
        ppo_agent.buffer_clear()

    pbar.close()
    if args.is_use_wandb:
        wandb.finish()

#############################################
##
#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # WANDB
    parser.add_argument('--is_use_wandb', default=False, action='store_true', help="[NOTICE] ")
    parser.add_argument('--wandb_project', type=str, default='reward_prediction_model', help="[NOTICE] ")
    parser.add_argument('--wandb_exp', type=str, default='test', help="[NOTICE] ")

    # SAVE
    parser.add_argument('--is_weight_save_best', default=False, action='store_true', help="[NOTICE] ")
    parser.add_argument('--is_weight_save', default=False, action='store_true',  help="[NOTICE] ")
    parser.add_argument('--weight_save_path', type=str, default='./weights/', help="[NOTICE] ")
    parser.add_argument('--weight_save_interval', type=int, default=5, help="[NOTICE] ")

    # LOAD
    parser.add_argument('--weight_load_path', type=str, default='./weights/hor/', help="[NOTICE] ")
    parser.add_argument('--weight_load_type', type=str, default='reward-pred', help="[NOTICE] ")
    parser.add_argument('--weight_load_tag', type=str, default='[sh-1]_[trj-30]_[epo-100]_[size-100]-[rwd-horizontal_position]', help="[NOTICE] ")

    # PRE-TRAIN 
    parser.add_argument('--dataset', type=str, default='Sprites', help="[NOTICE] ")
    parser.add_argument('--dataset_size', type=int, default=100, help="[NOTICE] ")
    parser.add_argument('--batch_size', type=int, default=1, help="[NOTICE] ")
    parser.add_argument('--max_epoch', type=int, default=100, help="[NOTICE] ")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="[NOTICE] ")
    parser.add_argument('--scheduler_gamma', type=float, default=0.99, help="[NOTICE] ")

    parser.add_argument('--input_channels', type=int, default=1, help="[NOTICE] ")
    
    parser.add_argument('--hidden_dim', type=int, default=64, help="[NOTICE] ")
    parser.add_argument('--mlp_hidden_units',type=int, default=32, help="[NOTICE] ")
    parser.add_argument('--lstm_num_layers', type=int, default=1, help="[NOTICE] ")
    parser.add_argument('--gpus', type=int, default=0, help="[NOTICE] ") # Not Supporting Multiple GPUs (GPUs = 1 or 0)
    
    # DATA
    parser.add_argument('--resolution', type=int, default=64, help="[NOTICE] ")
    parser.add_argument('--max_seq_len', type=int, default=30, help="[NOTICE] ")
    parser.add_argument('--max_cond_frame_len', type=int, default=5, help="[NOTICE] ")
    parser.add_argument('--max_speed', type=float, default=0.05, help="[NOTICE] ")
    parser.add_argument('--obj_size', type=float, default=0.2, help="[NOTICE] ")
    parser.add_argument('--shapes_per_traj', type=int, default=2, help="[NOTICE] ") # assume " [ AGENT(1) | TARGET(0) | Distractors(...)] "
    parser.add_argument('--rewards', nargs='+', type=str, default=['agent_y','agent_x','target_x','target_y'], help='[NOTICE] ')

    # PPO Parameters
    parser.add_argument('--num_timestep_total', type=int, default=5000000, help="[NOTICE] ")
    parser.add_argument('--num_timestep_per_batch', type=int, default=2048, help="[NOTICE] ")
    parser.add_argument('--num_timestep_per_episode', type=int, default=40, help="[NOTICE] ")
    parser.add_argument('--num_update_per_batch', type=int, default=10, help="[NOTICE] ")
    parser.add_argument('--normalization_bias', type=float, default=1e-9, help="[NOTICE] ")

    parser.add_argument('--r_gamma', type=float, default=0.95, help="[NOTICE] ")
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help="[NOTICE] ")
    parser.add_argument('--coef_value_loss', type=float, default=0.5, help="[NOTICE] ")
    parser.add_argument('--coef_entropy_loss', type=float, default=0.01, help="[NOTICE] ")
    parser.add_argument('--learning_rate_ppo', type=float, default=3e-4, help="[NOTICE] ")
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help="[NOTICE] ")

    # AC
    parser.add_argument('--latent_size_net', type=int, default=32, help="[NOTICE] ") 
    parser.add_argument('--input_resolution', type=int, default=64, help="[NOTICE] ") 
    parser.add_argument('--cnn_latent_channels', type=int, default=16, help="[NOTICE] ") 
    parser.add_argument('--output_channels', type=int, default=64, help="[NOTICE] the output channels of the encoder in Actor-Critic") 

    parser.add_argument('--output_size_policy', type=int, default=64, help="[NOTICE] ") 
    parser.add_argument('--output_size_value', type=int, default=1, help="[NOTICE] ") 

    parser.add_argument('--reward_w_path', type=str, default='', help="[NOTICE] ") 
    parser.add_argument('--reconstruction_w_path', type=str, default='', help="[NOTICE] ") 

    parser.add_argument('--is_visual_traj', default=False, action='store_true', help="[NOTICE] ")
    
    # CHOSE
    parser.add_argument('--num_distractors', type=int, default=0, help="[NOTICE] ")
    parser.add_argument('--mode', type=str, default='oracle', help="[NOTICE] ")
    
    # ======= MODE =======
    # - 'oracle'
    # - 'cnn'
    # - 'image_scratch', 
    # - 'image_reconstruction'
    # - 'image_reconstruction_finetune', 
    # - 'reward_prediction'
    # - 'reward_prediction_finetune'

    # ======= DATASET =======
    # - 'Sprites-v0'
    # - 'Sprites-v1'
    # - 'Sprites-v2'
    # - 'SpritesState-v0'
    # - 'SpritesState-v1'
    # - 'SpritesState-v2'

    args = parser.parse_args()

    # CHANGE
    if args.mode == 'cnn':
        args.latent_size_net = 64
    else:
        args.latent_size_net = 32

    # SHOW
    
    # print('[INFO] =================== [args] =================== ')
    # print(args)
    # print('[INFO] =================== [args] =================== ')
    # print('[INFO] Begin to train...')

    train_ppo(args)
