from pathlib import Path
import numpy as np
import os
import ppo2 as network
import torch
import gymnasium as gym
from env import ABREnv
from test_xatuRL import main
import matplotlib.pyplot as plt
import time
import logging
import sys


# Set up logging configuration
BASE_DIR = Path(__file__).resolve().parent
log_dir = BASE_DIR / 'logs'
log_dir.mkdir(exist_ok=True)
log_file_path = log_dir / 'xatu_training_log.txt'

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler(log_file_path),
                              logging.StreamHandler(sys.stdout)])

S_DIM = [7, 8]
A_DIM = 6
ACTOR_LR_RATE = 1e-4
TRAIN_SEQ_LEN = 160  # take as a train batch
TRAIN_EPOCH = 1500000 
MODEL_SAVE_INTERVAL = 50000 
RANDOM_SEED = 42
MODEL_DIR = str(BASE_DIR / 'models')
PLOTS_DIR = BASE_DIR / 'plots'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

env = ABREnv()
actor = network.Network(state_dim=S_DIM, action_dim=A_DIM,
                            learning_rate=ACTOR_LR_RATE)


tested_rewards = []
epochs_tested = []
overall_entropy = np.zeros([TRAIN_EPOCH+1, TRAIN_SEQ_LEN])
score = np.zeros([TRAIN_EPOCH+1])

logging.info("Starting PENSIVE TRAINING...")

for epoch in range(1, TRAIN_EPOCH+1):
    obs, info = env.reset()
    s_batch, a_batch, p_batch, r_batch = [], [], [], []
    
    for step in range(TRAIN_SEQ_LEN):
        s_batch.append(obs)
        action_prob = actor.predict(
                    np.reshape(obs, (1, S_DIM[0], S_DIM[1])))
        noise = np.random.gumbel(size=len(action_prob))
        bit_rate = np.argmax(np.log(action_prob) + noise)
        obs, rew, done, _, info = env.step(bit_rate)
        score[epoch] += rew
        logging.info(f"epoch: {epoch} | step: {step} | rew: {rew:.3f} | score: {score[epoch]:.3f} | entropy: {actor._entropy_weight:.3f}")
        logging.info(str(info))
        logging.info("-" * 200)
        # time.sleep(0.1)
        overall_entropy[epoch, step] = actor._entropy_weight

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1
        a_batch.append(action_vec)
        r_batch.append(rew)
        p_batch.append(action_prob)
        if done:
            break

    v_batch = actor.compute_v(s_batch, a_batch, r_batch, done)
    s_batch = np.stack(s_batch, axis=0)
    a_batch = np.vstack(a_batch)
    p_batch = np.vstack(p_batch)
    v_batch = np.vstack(v_batch)
    actor.train(s_batch, a_batch, p_batch, v_batch, epoch)

    if epoch % MODEL_SAVE_INTERVAL == 0:
        # Step 1: Save the model with a placeholder for test score
        placeholder_path = os.path.join(MODEL_DIR, f'xatu_model_ep_{epoch}_temp.pth')
        actor.save_model(placeholder_path)
        logging.info(f"Model initially saved at: {placeholder_path}")

        # Step 2: Compute the test score
        test_score = main(placeholder_path)
        tested_rewards.append(test_score)
        epochs_tested.append(epoch)
        logging.info(f"Test score after epoch {epoch}: {test_score:.3f}")

        # Step 3: Rename the file to include the test score
        final_path = os.path.join(MODEL_DIR, f'xatu_model_ep_{epoch}_{test_score:.3f}.pth')
        os.rename(placeholder_path, final_path)
        logging.info(f"Model file renamed to: {final_path}")


# Plot for Score vs Epoch
plt.figure(figsize=(12, 6))
plt.plot(epochs_tested, tested_rewards, label='Tests Overall Score', color='b')
plt.title('Score vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.grid(True)
plt.legend()
plt.savefig(PLOTS_DIR / 'score_vs_epoch.png', format='png')
plt.close()

# Plot for Entropy vs Epoch (mean entropy across steps in each epoch)
mean_entropy_per_epoch = np.mean(overall_entropy, axis=1)
plt.figure(figsize=(12, 6))
plt.plot(range(1, TRAIN_EPOCH+1), mean_entropy_per_epoch[1:TRAIN_EPOCH+1], label='Mean Entropy', color='r')
plt.title('Entropy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Entropy')
plt.grid(True)
plt.legend()
plt.savefig(PLOTS_DIR / 'entropy_vs_epoch.png', format='png')
plt.close()