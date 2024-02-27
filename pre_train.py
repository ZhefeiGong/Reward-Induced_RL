#@time   : Feb, 2024 
#@func   : 
#@author : Zhefei Gong

import wandb
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from model import MODEL_REWARD_PRD
from sprites_datagen.moving_sprites import MovingSpriteDataset
from general_utils import AttrDict
from torch.utils.data import DataLoader
from sprites_datagen.rewards import *

#############################################
##
#############################################
REWARDS = {
    'zero': ZeroReward,
    'vertical_position': VertPosReward, 
    'horizontal_position': HorPosReward,
    'agent_x': AgentXReward, 
    'agent_y': AgentYReward,
    'target_x': TargetXReward, 
    'target_y': TargetYReward,
}

#############################################
##@func      : 
##@author    : Zhefei Gong
#############################################
def train_reward_prediction(args):

    if args.use_wandb:
        wandb.init(project=args.wandb_project, 
                name=args.wandb_exp,
                config=args)

    _rewards = [REWARDS[reward] for reward in args.rewards] if args.rewards is not None else [ZeroReward]

    model = MODEL_REWARD_PRD(input_resolution=args.resolution,
                             input_channels=args.input_channels,
                             hidden_dim = args.hidden_dim,
                             rewards=_rewards)
    
    criterion = nn.MSELoss()

    optimizer = optim.RAdam(model.parameters(), lr=args.learning_rate, betas=[0.9, 0.999])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)

    _spec = AttrDict(
        resolution=args.resolution,             # the resolution of images
        max_seq_len=args.max_seq_len,           # the length of the sequence
        max_speed=args.max_speed,               # total image range [0, 1]
        obj_size=args.obj_size,                 # size of objects, full images is 1.0
        dataset_size=args.dataset_size,         # the number of epoches 
        shapes_per_traj=args.shapes_per_traj,   # number of shapes per trajectory
        rewards=_rewards,                       # rewards
    )
    train_loader = DataLoader(dataset = MovingSpriteDataset(spec=_spec), batch_size = args.batch_size, shuffle=True)

    for epoch in range(args.max_epoch):  #      

        # ========================= TRAIN =========================
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                optimizer.zero_grad()

                # train with "batch_size=1" 

                # print("[shape]states: ", batch.states.shape)             # [T, N, 2]
                # print("[shape]shape_idxs: ", batch.shape_idxs.shape)     # [0 1 2 3 ...]
                # print("[shape]images : ", batch.images.shape)            # [0 255]
                # print("[shape]rewards : ", batch.rewards[_rewards[0].NAME].shape)  # rewards

                x = torch.squeeze(batch.images) # 
                rewards_gt = torch.stack([torch.squeeze(batch.rewards[reward.NAME]) for reward in _rewards])    

                rewards_pred = model(x)

                loss = criterion(input = rewards_pred, target=rewards_gt)

                loss.backward()
                
                optimizer.step()

                # wandb
                if args.use_wandb:
                    wandb.log({"train_loss": loss.item()})
                
                # tqdm
                tepoch.set_postfix(loss=loss.item())
            
            lr_scheduler.step()

        # ========================= EVAL =========================
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0

            if args.use_wandb:
                wandb.log({"valid_loss": valid_loss})
    
        # ========================= WEIGHTS =========================
        if epoch % args.weight_save_interval == 0 and epoch != 0:
            path = args.weight_save_path + "reward_pred_model-epoch-"+str(epoch) + ".pth"
            print('[WEIGHT] save the weights in', path)
            torch.save(model.state_dict(), path) 
    
    if args.use_wandb:
        wandb.finish()


#############################################
##@func      : 
##@author    : Zhefei Gong
#############################################
def train_image_reconstruction():
    pass


#############################################
##
#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Sprites', help="[NOTICE] ")
    parser.add_argument('--dataset_size', type=int, default=100, help="[NOTICE] ")
    parser.add_argument('--max_epoch', type=int, default=1000, help="[NOTICE] ")

    parser.add_argument('--wandb_project', type=str, default='reward_prediction_model', help="[NOTICE] ")
    parser.add_argument('--wandb_exp', type=str, default='test', help="[NOTICE] ")

    parser.add_argument('--weight_save_path', type=str, default='./weights/', help="[NOTICE] ")
    parser.add_argument('--weight_save_interval', type=int, default=5, help="[NOTICE] ")

    parser.add_argument('--batch_size', type=int, default=1, help="[NOTICE] ")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="[NOTICE] ")
    parser.add_argument('--scheduler_gamma', type=float, default=0.99, help="[NOTICE] ")

    parser.add_argument('--resolution', type=int, default=64, help="[NOTICE] ")
    parser.add_argument('--max_seq_len', type=int, default=30, help="[NOTICE] ")
    parser.add_argument('--max_speed', type=float, default=0.05, help="[NOTICE] ")
    parser.add_argument('--obj_size', type=float, default=0.2, help="[NOTICE] ")

    parser.add_argument('--shapes_per_traj', type=int, default=2, help="[NOTICE] ") # assume " [ AGENT(1) | TARGET(0) | Distractors(...)] "
    
    parser.add_argument('--rewards', nargs='+', type=str, default=['agent_y','agent_x','target_x','target_y'], help='[NOTICE] ')

    parser.add_argument('--input_channels', type=int, default=3, help="[NOTICE] ")
    parser.add_argument('--hidden_dim', type=int, default=64, help="[NOTICE] ")
    
    parser.add_argument('--gpus', type=int, default=0, help="[NOTICE] ")
    parser.add_argument('--use_wandb', type=bool, default=True, help="[NOTICE] ")

    # parser.add_argument('--N', type=int, default=5, help='initial time window size') ？？？

    args = parser.parse_args()
    


    print('[INFO] =================== [args] =================== ')
    print(args)
    print('[INFO] =================== [args] =================== ')
    print('[INFO] Begin to train...')

    train_reward_prediction(args)

