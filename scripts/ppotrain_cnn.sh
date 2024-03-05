python ./ppo_train.py \
--mode 'cnn' \
--num_distractors 1 \
--wandb_project 'baseline-2M-1dtr' \
--wandb_exp 'cnn-run1' \
--num_timestep_total 2000000 \
--num_timestep_per_batch 4000 \
--num_timestep_per_episode 40 \
--num_update_per_batch 10 \
--normalization_bias 1e-9 \
--latent_size 64 \
--r_gamma 0.95 \
--clip_epsilon 0.2 \
--coef_value_loss 0.5 \
--coef_entropy_loss 0.01 \
--learning_rate_ppo 3e-4 \
--max_grad_norm 0.5 \
--latent_size_net 64 \
--input_resolution 64 \
--cnn_latent_channels 16 \
--output_channels 64 \
--output_size_policy 64 \
--output_size_value 1 \
--reward_w_path '' \
--reconstruction_w_path '' \
--gpus_num 1 \
--gpus_idx 5 \
--is_use_wandb \

# --num_timestep_total 5000000 \
# --num_timestep_per_batch 2048 \
# --is_weight_save \
# --is_weight_save_best \

