python ./ppo_train.py \
--mode 'reward_prediction' \
--num_distractors 1 \
--wandb_project 'baseline-2M-1dtr' \
--wandb_exp 'reward_prediction-run1' \
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
--latent_size_net 32 \
--input_resolution 64 \
--cnn_latent_channels 16 \
--output_channels 64 \
--output_size_policy 64 \
--output_size_value 1 \
--reward_w_path 'weights/fig3/rwd_xyxy_1dtr_500/reward-pred_encoder_[sh-3]_[trj-40]_[cf-5]_[epo-500]_[size-100]-[rwd-agent_x-agent_y-target_x-target_y].pth' \
--reconstruction_w_path '' \
--gpus_num 1 \
--gpus_idx 7 \
--is_use_wandb \

# --is_visual_traj \


# --reward_w_path 'weights/fig3/rwd_xyxy_1dtr_500/reward-pred_encoder_[sh-3]_[trj-40]_[cf-5]_[epo-500]_[size-100]-[rwd-agent_x-agent_y-target_x-target_y].pth' \

# --reward_w_path 'weights/fig3/rwd_xyxy_1/reward-pred_encoder_[sh-2]_[trj-40]_[cf-5]_[epo-1]_[size-100]-[rwd-agent_x-agent_y-target_x-target_y].pth' \
# --reward_w_path 'weights/fig3/rwd_xyxy_50/reward-pred_encoder_[sh-2]_[trj-40]_[cf-5]_[epo-50]_[size-100]-[rwd-agent_x-agent_y-target_x-target_y].pth' \
# --reward_w_path 'weights/fig3/rwd_xyxy_100/reward-pred_encoder_[sh-2]_[trj-40]_[cf-5]_[epo-100]_[size-100]-[rwd-agent_x-agent_y-target_x-target_y].pth' \
# --reward_w_path 'weights/fig3/rwd_xyxy_200/reward-pred_encoder_[sh-2]_[trj-40]_[cf-5]_[epo-200]_[size-100]-[rwd-agent_x-agent_y-target_x-target_y].pth' \
# --reward_w_path 'weights/fig3/rwd_xyxy_500/reward-pred_encoder_[sh-2]_[trj-40]_[cf-5]_[epo-500]_[size-100]-[rwd-agent_x-agent_y-target_x-target_y].pth' \

# --num_timestep_total 5000000 \
# --num_timestep_per_batch 2048 \
# --is_weight_save \
# --is_weight_save_best \

