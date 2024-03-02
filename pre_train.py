#@time   : Feb, 2024 
#@func   : 
#@author : Zhefei Gong

import wandb
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from model import MODEL_REWARD_PRD,ENCODER,DECODER,LSTM,MLPs
from sprites_datagen.moving_sprites import MovingSpriteDataset
from general_utils import AttrDict,make_figure2
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
def train_reward_prediction_model(args):
    # WANDB
    if args.is_use_wandb:
        wandb.init(project=args.wandb_project, 
                name=args.wandb_exp,
                config=args)

    # MODEL
    _rewards = [REWARDS[reward] for reward in args.rewards] if args.rewards is not None else [ZeroReward]
    model = MODEL_REWARD_PRD(input_resolution=args.resolution,
                             input_channels=args.input_channels,
                             hidden_dim = args.hidden_dim,
                             mlp_hidden_units = args.mlp_hidden_units,
                             lstm_num_layers = args.lstm_num_layers,
                             rewards=_rewards)
    model.w_init() # init with kaiming
    
    # CRITERIA
    criterion = nn.MSELoss()
    optimizer = optim.RAdam(model.parameters(), lr=args.learning_rate, betas=[0.9, 0.999])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)

    # DATASET
    _spec = AttrDict(
        resolution=args.resolution,             # the resolution of images
        max_seq_len=args.max_seq_len,           # the length of the sequence
        max_speed=args.max_speed,               # total image range [0, 1]
        obj_size=args.obj_size,                 # size of objects, full images is 1.0
        dataset_size=args.dataset_size,         # the number of epoches 
        shapes_per_traj=args.shapes_per_traj,   # number of shapes per trajectory
        rewards=_rewards,                       # rewards
        input_channels = args.input_channels,   #
    )
    train_loader = DataLoader(dataset = MovingSpriteDataset(spec=_spec), batch_size = args.batch_size, shuffle=True)

    # RUN - [*] train with "batch_size=1" [*]
    epoch_loss_min = float("inf")
    for epoch in range(args.max_epoch):  #      

        # ========================= TRAIN =========================
        epoch_loss = 0
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                optimizer.zero_grad()

                # print("[shape]states: ", batch.states.shape)             # [T, N, 2]
                # print("[shape]shape_idxs: ", batch.shape_idxs.shape)     # [0 1 2 3 ...]
                # print("[shape]images : ", batch.images.shape)            # [0 255]
                # print("[shape]rewards : ", batch.rewards[_rewards[0].NAME].shape)  # rewards

                x = torch.squeeze(batch.images, dim=0) # [1, T, C, H, W] -> [T, C, H, W]
                # [**] Use the first N frames [**]cle   
                x = x[:args.max_cond_frame_len, :, :, :] # [T(0-N~T), C, H, W]
                # [**] Make other frames as zero [**]
                x = torch.cat([x, torch.zeros((args.max_seq_len - args.max_cond_frame_len, x.size()[1], x.size()[2], x.size()[3]))])
                
                rewards_gt = torch.squeeze(torch.stack([torch.squeeze(batch.rewards[reward.NAME]) for reward in _rewards]))  # [N, T]
                
                rewards_pred = model(x)

                loss = criterion(input=rewards_pred, target=rewards_gt)

                loss.backward()
                
                optimizer.step()

                epoch_loss += loss.item()

                # wandb
                if args.is_use_wandb:
                    wandb.log({"train_loss": loss.item()})
                
                # tqdm
                tepoch.set_postfix(loss=loss.item())
            
            lr_scheduler.step()

        # ========================= EVAL =========================
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0

            if args.is_use_wandb:
                wandb.log({"valid_loss": valid_loss})
    
        # ========================= SAVE =========================
        if args.is_weight_save :
            
            reward_tag = '-'.join(args.rewards)
            train_tag = f"[sh-{args.shapes_per_traj}]_[trj-{args.max_seq_len}]_[epo-{args.max_epoch}]_[size-{args.dataset_size}]-[rwd-{reward_tag}].pth"

            # save the best weight
            if args.is_weight_save_best:
                if epoch_loss < epoch_loss_min:
                    epoch_loss_min = epoch_loss
                    print('[SAVE] save the weights as : ', train_tag, f'[EPOCH-{epoch} | LOSS-{epoch_loss_min}')
                    torch.save(model.encoder.state_dict(), args.weight_save_path+"reward-pred_encoder_"+train_tag)
                    torch.save(model.mlp.state_dict(), args.weight_save_path+"reward-pred_mlp_"+train_tag)
                    torch.save(model.lstm.state_dict(), args.weight_save_path+"reward-pred_lstm_"+train_tag)    
            
            # save the weight with training process
            elif epoch % args.weight_save_interval == 0 and epoch != 0:
                print('[SAVE] save the weights as : ', train_tag, f'[EPOCH-{epoch} | LOSS-{epoch_loss}')
                torch.save(model.encoder.state_dict(), args.weight_save_path+f"reward-pred_encoder[{epoch}]_"+train_tag)
                torch.save(model.mlp.state_dict(), args.weight_save_path+f"reward-pred_mlp[{epoch}]_"+train_tag)
                torch.save(model.lstm.state_dict(), args.weight_save_path+f"reward-pred_lstm[{epoch}]_"+train_tag) 
    
    if args.is_use_wandb:
        wandb.finish()

#############################################
##@func      : 
##@author    : Zhefei Gong
#############################################
def train_image_reconstruction_decoder(args):

    if args.is_use_wandb:
        wandb.init(project=args.wandb_project, 
                name=args.wandb_exp,
                config=args)

    _rewards = [REWARDS[reward] for reward in args.rewards] if args.rewards is not None else [ZeroReward]

    # MODEL
    encoder = ENCODER(input_resolution = args.resolution,
                      input_channels = args.input_channels, 
                      output_channels = args.hidden_dim)
    mlp = MLPs(input_size = args.hidden_dim, 
               output_size = args.hidden_dim, 
               hidden_units = args.mlp_hidden_units)
    lstm = LSTM(input_size = args.hidden_dim,
                hidden_size = args.hidden_dim,
                 num_layers = args.lstm_num_layers)
    decoder = DECODER(input_channels = args.hidden_dim, 
                      output_resolution = args.resolution, 
                      output_channels = args.input_channels) # 

    # INIT
    encoder.load_state_dict(torch.load(f"{args.weight_load_path}{args.weight_load_type}_encoder_{args.weight_load_tag}.pth"))
    mlp.load_state_dict(torch.load(f"{args.weight_load_path}{args.weight_load_type}_mlp_{args.weight_load_tag}.pth"))
    lstm.load_state_dict(torch.load(f"{args.weight_load_path}{args.weight_load_type}_lstm_{args.weight_load_tag}.pth"))
    decoder.w_init() # USE He Kaiming Initialization Method 
    
    # STATE -> Can't stop autograd
    encoder.eval()
    mlp.eval()
    lstm.eval()
    decoder.train()

    # CRITERIA
    criterion = nn.MSELoss()
    optimizer = optim.RAdam(decoder.parameters(), lr=args.learning_rate, betas=[0.9, 0.999])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)

    _spec_train = AttrDict(
        resolution=args.resolution,             # the resolution of images
        max_seq_len=args.max_seq_len,           # the length of the sequence
        max_speed=args.max_speed,               # total image range [0, 1]
        obj_size=args.obj_size,                 # size of objects, full images is 1.0
        dataset_size=args.dataset_size,         # the number of trajectories
        shapes_per_traj=args.shapes_per_traj,   # number of shapes per trajectory
        rewards=_rewards,                       # rewards
        input_channels = args.input_channels,   #
    )
    _spec_eval = AttrDict(
        resolution=args.resolution,             # the resolution of images
        max_seq_len=args.max_seq_len,           # the length of the sequence
        max_speed=args.max_speed,               # total image range [0, 1]
        obj_size=args.obj_size,                 # size of objects, full images is 1.0
        dataset_size=args.max_epoch+1,          # the number of trajectories
        shapes_per_traj=args.shapes_per_traj,   # number of shapes per trajectory
        rewards=_rewards,                       # rewards
        input_channels = args.input_channels,   #
    )
    train_dataloader = DataLoader(dataset = MovingSpriteDataset(spec=_spec_train), batch_size = args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset = MovingSpriteDataset(spec=_spec_eval), batch_size = args.batch_size, shuffle=True)
    eval_dataloader_iter = iter(eval_dataloader)

    # RUN
    epoch_loss_min = float("inf")
    for epoch in range(args.max_epoch):  #      

        # ========================= TRAIN =========================
        epoch_loss = 0
        decoder.train()

        # [*[*[Assume train with "batch_size=1"]*]*]
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                optimizer.zero_grad()

                # Part-Data
                images_gt = torch.squeeze(batch.images, dim=0) # [1, T, C, H, W] -> [T, C, H, W]
                # [**] Use the first N frames [**]
                x = images_gt[:args.max_cond_frame_len, :, :, :] # [T(0-N~T), C, H, W]
                # [**] Make other frames as zero [**]
                x = torch.cat([x, torch.zeros((args.max_seq_len - args.max_cond_frame_len, x.size()[1], x.size()[2], x.size()[3]))])

                out = encoder(x) # [T,C,H,W] -> [T, dim]
                out = mlp(out)  # [T, dim] -> [T, dim]
                out = lstm(out) # [T, dim] -> [T, dim]
                
                out_detached = out.detach()
                
                images_recon = decoder(out_detached) # [T, dim] -> [T,C,H,W]

                loss = criterion(input=images_recon, target=images_gt)
                
                loss.backward()
                
                optimizer.step()

                epoch_loss += loss.item()

                # wandb
                if args.is_use_wandb:
                    wandb.log({"train_loss": loss.item()})
                
                # tqdm
                tepoch.set_postfix(loss=loss.item())
            
            lr_scheduler.step()

        # ========================= EVAL =========================
        decoder.eval()
        with torch.no_grad():
            batch = next(eval_dataloader_iter)

            images_gt = torch.squeeze(batch.images, dim=0) # [1, T, C, H, W] -> [T, C, H, W]
            
            # [**] Use the first N frames [**]
            x = images_gt[:args.max_cond_frame_len, :, :, :] # [T(0-N~T), C, H, W]
            # [**] Make other frames as zero [**]
            x = torch.cat([x, torch.zeros((args.max_seq_len - args.max_cond_frame_len, x.size()[1], x.size()[2], x.size()[3]))])
            
            out = encoder(x) # [T,C,H,W] -> [T, dim]
            out = mlp(out)  # [T, dim] -> [T, dim]
            out = lstm(out) # [T, dim] -> [T, dim]

            out_detached = out.detach() # Detached
            
            images_recon = decoder(out_detached) # [T, dim] -> [T,C,H,W]
            
            valid_loss = criterion(input=images_gt, target=images_recon)

            if args.is_use_wandb:
                fig2, idxs = make_figure2(imgs_gt = images_gt, imgs_pred = images_recon)
                idxs_str = '-'.join(map(str, idxs))
                fig2 = np.transpose(fig2.numpy(), (1,2,0))
                Img = wandb.Image(fig2, caption="epoch:{} <--> {}".format(epoch, idxs_str))
                wandb.log({"Figure 2 " : Img})
                wandb.log({"valid_loss": valid_loss.item()})

        # ========================= SAVE =========================
        if args.is_weight_save :
            
            reward_tag = '-'.join(args.rewards)
            train_tag = f"[sh-{args.shapes_per_traj}]_[trj-{args.max_seq_len}]_[epo-{args.max_epoch}]_[size-{args.dataset_size}]-[rwd-{reward_tag}].pth"

            # save the best weight
            if args.is_weight_save_best:
                if epoch_loss < epoch_loss_min:
                    epoch_loss_min = epoch_loss
                    print('[SAVE] save the weights as : ', train_tag, f'[EPOCH-{epoch} | LOSS-{epoch_loss_min}')
                    torch.save(decoder.state_dict(), args.weight_save_path+"image-recon_decoder_"+train_tag)

            # save the weight with training process
            elif epoch % args.weight_save_interval == 0 and epoch != 0:
                print('[SAVE] save the weights as : ', train_tag, f'[EPOCH-{epoch} | LOSS-{epoch_loss}')
                torch.save(decoder.state_dict(), args.weight_save_path+f"image-recon_decoder[{epoch}]_"+train_tag)
         
    if args.is_use_wandb:
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

    # MODE
    parser.add_argument('--mode', type=str, default='reward_prediction_model', help="[NOTICE] ")

    # WANDB
    parser.add_argument('--is_use_wandb', default=False, action='store_true', help="[NOTICE] ")
    parser.add_argument('--wandb_project', type=str, default='reward_prediction_model', help="[NOTICE] ")
    parser.add_argument('--wandb_exp', type=str, default='test', help="[NOTICE] ")

    # SAVE
    parser.add_argument('--weight_save_path', type=str, default='./weights/', help="[NOTICE] ")
    parser.add_argument('--weight_save_interval', type=int, default=5, help="[NOTICE] ")
    parser.add_argument('--is_weight_save_best', default=False, action='store_true', help="[NOTICE] ")
    parser.add_argument('--is_weight_save', default=False, action='store_true',  help="[NOTICE] ")

    # LOAD
    parser.add_argument('--weight_load_path', type=str, default='./weights/hor/', help="[NOTICE] ")
    parser.add_argument('--weight_load_type', type=str, default='reward-pred', help="[NOTICE] ")
    parser.add_argument('--weight_load_tag', type=str, default='[sh-1]_[trj-30]_[epo-100]_[size-100]-[rwd-horizontal_position]', help="[NOTICE] ")

    # TRAIN
    parser.add_argument('--dataset', type=str, default='Sprites', help="[NOTICE] ")
    parser.add_argument('--dataset_size', type=int, default=100, help="[NOTICE] ")
    parser.add_argument('--batch_size', type=int, default=1, help="[NOTICE] ")
    parser.add_argument('--max_epoch', type=int, default=100, help="[NOTICE] ")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="[NOTICE] ")
    parser.add_argument('--scheduler_gamma', type=float, default=0.99, help="[NOTICE] ")
    parser.add_argument('--input_channels', type=int, default=3, help="[NOTICE] ")
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


    args = parser.parse_args()

    print('[INFO] =================== [args] =================== ')
    print(args)
    print('[INFO] =================== [args] =================== ')
    print('[INFO] Begin to train...')

    if args.mode == 'reward_prediction_model':
        train_reward_prediction_model(args)
    elif args.mode == 'image_reconstruction_decoder':
        train_image_reconstruction_decoder(args)
