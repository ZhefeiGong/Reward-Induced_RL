#@time   : Mar, 2024 
#@func   : 
#@author : Zhefei Gong


import wandb

import gym
import torch
import torch.nn as nn
import argparse
from general_utils import AttrDict
from ppo import MODEL_PPO


#@func : 
def train_ppo(args):

    # PARA
    num_timestep_total = args.num_timestep_total
    num_update_per_batch = args.num_update_per_batch
    learning_rate_ppo = args.learning_rate_ppo
    max_grad_norm = args.max_grad_norm

    # INIT
    count_timestep_total = 0

    # ENV
    env_id = f"{'SpritesState' if args.baseline_name == 'oracle' else 'Sprites'}-v{args.num_distractors}"
    env = gym.make(env_id)

    # MODEL
    _spec = AttrDict(
        mode = args.mode,
        num_timestep_total = args.num_timestep_total,
        num_timestep_per_batch = args.num_timestep_per_batch,
        num_timestep_per_episode = args.num_timestep_per_episode,
        input_channels = args.input_channels,
        normalization_bias = args.normalization_bias,
        latent_size = args.latent_size,
        r_gamma = args.r_gamma,
        clip_epsilon = args.clip_epsilon,
        coef_value_loss = args.coef_value_loss,
        coef_entropy_loss = args.coef_entropy_loss,
    )
    ppo_agent = MODEL_PPO(env=env, spec=_spec)
    ppo_agent.w_init() # initialize the weights

    # OPTIMIZER
    optimizer = torch.optim.Adam(ppo_agent.agent.parameters(), lr=learning_rate_ppo)

    # TRAIN
    while count_timestep_total < num_timestep_total:
        
        # ACTOR
        with torch.no_grad():
            ppo_agent.rollout()
            ppo_agent.compute_RTGs()
            ppo_agent.compute_advantage_estimate
            count_timestep_total += ppo_agent.count_timestep_per_batch

        # CRITIC
        with torch.enable_grad():
            for _ in range(num_update_per_batch):
                optimizer.zero_grad()
                loss = ppo_agent.update()
                loss.backward()
                nn.utils.clip_grad_norm_(ppo_agent.agent.parameters(), max_grad_norm)
                optimizer.step()
        
        # CLEAR
        ppo_agent.buffer_clear()


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
    parser.add_argument('--gpus', type=int, default=0, help="[NOTICE] ")
    
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
    parser.add_argument('--normalization_bias', type=int, default=1e-9, help="[NOTICE] ")
    parser.add_argument('--latent_size', type=int, default=64, help="[NOTICE] ")
    parser.add_argument('--r_gamma', type=float, default=0.95, help="[NOTICE] ")
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help="[NOTICE] ")
    parser.add_argument('--coef_value_loss', type=float, default=0.5, help="[NOTICE] ")
    parser.add_argument('--coef_entropy_loss', type=float, default=0.01, help="[NOTICE] ")
    parser.add_argument('--learning_rate_ppo', type=float, default=3e-4, help="[NOTICE] ")
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help="[NOTICE] ")

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

    print('[INFO] =================== [args] =================== ')
    print(args)
    print('[INFO] =================== [args] =================== ')
    print('[INFO] Begin to train...')

    train_ppo(args)
