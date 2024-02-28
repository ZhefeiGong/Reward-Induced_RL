python ./pre_train.py \
--mode 'image_reconstruction_decoder' \
--wandb_project 'image_reconstruction_decoder' \
--wandb_exp 'run8-part-data-1c' \
--weight_save_path './weights/dec/' \
--weight_save_interval 5 \
--dataset 'Sprites' \
--dataset_size 50 \
--batch_size 1 \
--max_epoch 100 \
--learning_rate 1e-3 \
--scheduler_gamma 0.99 \
--input_channels 1 \
--hidden_dim 64 \
--mlp_hidden_units 32 \
--gpus 0 \
--resolution 64 \
--max_seq_len 30 \
--max_cond_frame_len 5 \
--max_speed 0.05 \
--obj_size 0.2 \
--shapes_per_traj 1 \
--rewards 'horizontal_position' \
--weight_load_path './weights/hor_1c/' \
--weight_load_type 'reward-pred' \
--weight_load_tag '[sh-1]_[trj-30]_[epo-500]_[size-200]-[rwd-horizontal_position]' \
--is_use_wandb \

# --is_weight_save \
# --is_weight_save_best \

# --rewards 'agent_x' 'agent_y' 'target_x' 'target_y' \
