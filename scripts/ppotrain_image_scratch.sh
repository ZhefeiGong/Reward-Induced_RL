python ./ppo_train.py \
--mode 'image_scratch' \
--num_distractors 0 \
--wandb_project 'baseline' \
--wandb_exp 'image_scratch-run1' \
--num_timestep_total 50000 \
--num_timestep_per_batch 400 \
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
--latent_size_net 32 \
--input_resolution 64 \
--cnn_latent_channels 16 \
--output_channels 64 \
--output_size_policy 64 \
--output_size_value 1 \
--reward_w_path '' \
--reconstruction_w_path '' \
--is_use_wandb \

# --num_timestep_total 5000000 \
# --num_timestep_per_batch 2048 \
# --is_weight_save \
# --is_weight_save_best \
