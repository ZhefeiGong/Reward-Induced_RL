import cv2
from general_utils import make_image_seq_strip
from sprites_datagen.rewards import *
import numpy as np
import cv2

from general_utils import AttrDict, make_gif
from sprites_datagen.moving_sprites import DistractorTemplateMovingSpritesGenerator,TemplateMovingSpritesGenerator
from sprites_env.envs.sprites import SpritesEnv


#@func : test the dataset of sprites
def TEST_sprites_dataset():
    spec = AttrDict(
        resolution=128,             # 
        max_seq_len=30,             # the length of the sequence
        max_speed=0.05,             # total image range [0, 1]
        obj_size=0.2,               # size of objects, full images is 1.0
        shapes_per_traj=1,          # number of shapes per trajectory
        rewards=[ZeroReward, AgentXReward],
    )
    gen = DistractorTemplateMovingSpritesGenerator(spec)
    if spec.shapes_per_traj == 1:
        gen = TemplateMovingSpritesGenerator(spec)
    traj = gen.gen_trajectory()
    img = make_image_seq_strip([traj.images[None, :, None].repeat(3, axis=2).astype(np.float32)], sep_val=255.0).astype(np.uint8)
    
    print("[shape]states: ", traj.states.shape)             # [T,N,2]
    print("[shape]shape_idxs: ", traj.shape_idxs.shape)     # [0 1 2 3 ...]
    print("[shape]images : ", traj.images.shape)            # [0 255]
    print("[shape]rewards : ", traj.rewards[ZeroReward.NAME].shape)  # rewards
    
    # print("shape_idxs : ", traj.shape_idxs)                 # 
    # print("states: ", traj.states[:,1,:])                   # 

    cv2.imwrite("tmp/test.png", img[0].transpose(1, 2, 0))

    print(traj.images.shape)

    make_gif(imgs = traj.images, path = "tmp/test.gif", fps_default=5)

#@func : test the environment of sprites
def TEST_sprites_env():
    data_spec = AttrDict(
        resolution=128,
        max_ep_len=40,
        max_speed=0.05,     # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        follow=True,
    )
    env = SpritesEnv()# kwarg['n_distractors'] choose the number of distractors (if not it's one)
    env.set_config(data_spec)

    obs = env.reset()
    cv2.imwrite("tmp/test_rl.png", 255 * np.expand_dims(obs, -1))

    # Control the Movement of the Agent by 
    # Passing in a 2D Array of X,Y-Velocities
    
    batch_obs = []
    for i in range(200):
        obs, reward, done, info = env.step([00.1, 0])
        batch_obs.append(obs)
        if done:
            print(reward)
    
    make_gif(imgs = np.array(batch_obs) * 255, path = "tmp/test_env.gif", fps_default=10)

    # print(obs.shape) # [64,64]
    # print(reward) # value
    # print(done) # True / False
    # print(info) # {}

    cv2.imwrite("tmp/test_rl_1.png", 255 * np.expand_dims(obs, -1))


if __name__ == "__main__":

    # TEST_sprites_dataset()

    TEST_sprites_env()
    
    # import gym
    # env = gym.make('Sprites-v1')
    # obs = env.reset()
    # cv2.imwrite("tmp/test_rl.png", 255 * np.expand_dims(obs, -1))
    # obs, reward, done, info = env.step([1, 0])
    # cv2.imwrite("tmp/test_rl_1.png", 255 * np.expand_dims(obs, -1))
