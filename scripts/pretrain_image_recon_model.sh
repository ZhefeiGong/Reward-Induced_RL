python ./pre_train.py \
--mode 'image_reconstruction_model' \
--wandb_project 'image_reconstruction_model' \
--wandb_exp 'run1-channel_1' \
--weight_save_path './weights/fig3/rec/' \
--resolution 64 \
--input_channels 1 \
--hidden_dim 64 \
--weight_save_interval 5 \
--dataset 'Sprites' \
--dataset_size 100 \
--max_epoch 200 \
--max_seq_len 40 \
--max_cond_frame_len 5 \
--batch_size 1 \
--learning_rate 1e-3 \
--scheduler_gamma 0.99 \
--mlp_hidden_units 32 \
--gpus 0 \
--max_speed 0.05 \
--obj_size 0.2 \
--shapes_per_traj 2 \
--rewards 'vertical_position' \

# --is_weight_save \
# --is_weight_save_best \
# --is_use_wandb \

# --rewards 'agent_x' 'agent_y' 'target_x' 'target_y' 'vertical_position' 'horizontal_position' \
