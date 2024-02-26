### Tasks : 
 1. Implement the reward-induced representation learning model (see detailed architecture Figure 5). Train the model using the provided dataset. 
 
 2. Visualize your training results (learning curves etc) and verify that the model predicts correctly by replicating the results in Figure 2. Note, that you need to add a detached decoder network and train the model on a single reward only to reproduce the experiment. 
 
 3. Implement an RL algorithm of your choice to train an agent that can follow a target shape (while ignoring distractor shapes) in the provided environment. Before continuing to the next step, verify your implementation by training a policy that has access to the true state of the environment (i.e. does not need to encode images). This corresponds to the oracle baseline in Figure 3. 
 
 4. Train an image-based agent that encodes image observations using the pre-trained encoder. Compare its learning curve to that of an agent with the same architecture, but trained from scratch (image-scratch baseline in Figure 3). 
 
 5. If your implementation is working, (1) the image-based agent with pre-training should be able to follow the target shape with up to one distractor, and (2) it should learn faster than the image-based agent trained from scratch (but likely slower than the oracle).


### INFO about DATA:

1. dataset - trajectory : 
    * states
        * the length of the sequence
        * number of shapes
        * the coordinate (x,y)
    * shape_idxs
        * [ AGENT(1) | TARGET(0) | Distractors(...)]
    * images
        * 👇 - x - vertical 
        * 👉 - y - horizontal 
    * rewards
        * value

2. environment :
    * just run :
    ```
    import gym
    env = gym.make('Sprites-v1')
    ``` 




