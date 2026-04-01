import numpy as np
import os
import ppo2 as network
import torch
import gymnasium as gym
from env import ABREnv
import time

S_DIM = [7, 8]
A_DIM = 6
ACTOR_LR_RATE = 1e-4
RANDOM_SEED = 42


def main(nn_model):
  
    env = ABREnv(train=False)
    actor = network.Network(state_dim=S_DIM, action_dim=A_DIM,
                            learning_rate=ACTOR_LR_RATE)

    # Load trained model
    actor.load_model(nn_model)
    print("Starting PENSIVE TESTING...")

    obs, info = env.reset()
    score = np.zeros([len(env.all_file_names)])
    video_count = 0

    while True:
    
        action_prob = actor.predict(
            np.reshape(obs, (1, S_DIM[0], S_DIM[1]))
        )
        bit_rate = np.argmax(np.log(action_prob))

        obs, rew, done, _, info = env.step(bit_rate)
        score[video_count] += rew

        print(f"File: {env.all_file_names[video_count]} | Chunks: {obs[6, -1]} | rew: {rew:.3f} | score: {score[video_count]:.3f}")
        print(info)
        print("-" * 200)
        # time.sleep(0.1)

        # Check if the video is finished
        if done:
            video_count += 1
            obs, info = env.reset()

            # If all videos have been processed, break the loop
            if video_count >= len(env.all_file_names):
                break

    total_score = np.sum(score)

    return total_score

# if __name__ == "__main__":
#     nn_model = "/path/to/your/model.pth" 
#     total_score = main(nn_model)
#     print(f"Total Score: {total_score:.3f}")