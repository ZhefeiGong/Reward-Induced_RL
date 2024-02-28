python ./pre_train.py \
--mode 'reward_prediction_model' \
--wandb_project 'reward_prediction_model' \
--wandb_exp 'run2-channel_1' \
--weight_save_path './weights/hor_1c/' \
--weight_save_interval 5 \
--dataset 'Sprites' \
--dataset_size 200 \
--batch_size 1 \
--max_epoch 500 \
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
--is_weight_save \
--is_weight_save_best \
--is_use_wandb \

# --rewards 'agent_x' 'agent_y' 'target_x' 'target_y' \
