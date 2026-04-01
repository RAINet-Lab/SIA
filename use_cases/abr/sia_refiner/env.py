# add queuing delay into halo
import os
import sys
from sia.core.constants import ADG_SKYNET_ROOT_ADDR
import numpy as np
import core as abrenv
import fixed_env
import load_trace
import gymnasium as gym
import torch
import torch.nn as nn
from Patch_TST import PatchTST

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 7
S_LEN = 8  # take how many frames in the past
A_DIM = 6
tput_forecast_window = 12
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = np.array([300., 750., 1200., 1850., 2850., 4300.])  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
EPS = 1e-6
TEST_TRACES = ADG_SKYNET_ROOT_ADDR + '/pensive/test_all_files/'  # Path to test traces
FORECASTER_PATH = ADG_SKYNET_ROOT_ADDR + '/pensive/pensive-gamma/TST-models/tst3_epoch_1200_loss_0.1210.pt'
n_token = 12           # lookback window size
input_dim = 1         # univariate input
model_dim = 64        # model/embedding dimension
num_heads = 4         # number of heads for multi-head attention
num_layers = 3        # number of transformer layers
output_dim = 4        # output dimension = horizon
FORECASTER_MODEL = PatchTST(n_token, input_dim, model_dim, num_heads, num_layers, output_dim)


class ABREnv(gym.Env):
    """
    Adaptive Bitrate (ABR) Streaming Environment based on OpenAI Gym.
    
    This environment simulates network conditions for streaming, allowing an agent to select video quality based on past and current network conditions.
    """

    def __init__(self, random_seed=RANDOM_SEED, train=True, TEST_TRACES = TEST_TRACES, FORECASTER_MODEL = FORECASTER_MODEL, FORECASTER_PATH = FORECASTER_PATH, if_MLP = False):
        
        """
        Initialize the ABR environment.

        Args:
            random_seed (int): Random seed for consistency.
            train (bool): Flag to indicate if in training mode.
            TEST_TRACES (str): Path to test network traces if in evaluation mode.
        """
        
        np.random.seed(random_seed)
        FORECASTER_criterion = nn.MSELoss() 
        print('Loading forecaster model from:', FORECASTER_PATH)
        FORECASTER_MODEL.load_state_dict(torch.load(FORECASTER_PATH, weights_only=True))
        FORECASTER_MODEL.eval()

        self.MLP = if_MLP
        if train:
            all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()
            self.net_env = abrenv.Environment(all_cooked_time=all_cooked_time,
                                              all_cooked_bw=all_cooked_bw,
                                              random_seed=random_seed)
        else:
            all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)
            self.all_file_names = all_file_names
            self.net_env = fixed_env.Environment(all_cooked_time=all_cooked_time,
                                          all_cooked_bw=all_cooked_bw,
                                          random_seed=random_seed)

        self.last_bit_rate = DEFAULT_QUALITY
        self.buffer_size = 0.
        self.state = np.zeros((S_INFO, S_LEN))
        self.state_bw = np.zeros(tput_forecast_window)
        self.time_stamp = 0
        self.action_space = gym.spaces.Discrete(A_DIM)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(S_INFO, S_LEN), dtype=np.float32)
        self.forecaster_model = FORECASTER_MODEL.to('cuda')
        
    def seed(self, num):
        np.random.seed(num)

    def reset(self):
        # self.net_env.reset_ptr()
        self.time_stamp = 0
        self.last_bit_rate = DEFAULT_QUALITY
        self.state = np.zeros((S_INFO, S_LEN))
        self.buffer_size = 0.
        bit_rate = self.last_bit_rate
        delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, curr_bw = \
            self.net_env.get_video_chunk(bit_rate)
        state = np.roll(self.state, -1, axis=1)
        state_bw = np.roll(self.state_bw, -1, axis=0)
        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state_bw[-1] = curr_bw
        if self.MLP:
            tput_pred = self.forecaster_model(torch.tensor(state_bw, dtype=torch.float32).to('cuda')).detach().cpu().numpy()
            netbw_pred = tput_pred
        else:
            tput_pred = self.forecaster_model(torch.tensor(state_bw, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to('cuda')).detach().cpu().numpy()
            netbw_pred = tput_pred.squeeze(0)
        tput_history = state_bw[-4:]
        state[4, :] = np.concatenate([tput_history, netbw_pred])
        state[5, :A_DIM] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[6, -1] = np.minimum(video_chunk_remain,
                                  CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        self.state = state
        self.state_bw = state_bw
        return state, {'bitrate': VIDEO_BIT_RATE[bit_rate], 'rebuffer': rebuf}

    def render(self):
        return

    def step(self, action):
        """
        Executes one time-step in the environment with the given action.

        Args:
            action (int): Selected bitrate level by the agent.

        Returns:
            Tuple: Next state, reward, done flag (end of video), truncated flag, and additional info.
        """

        bit_rate = int(action)
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, curr_bw = \
            self.net_env.get_video_chunk(bit_rate)
        
        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smooth penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                      VIDEO_BIT_RATE[self.last_bit_rate]) / M_IN_K

        self.last_bit_rate = bit_rate
        state = np.roll(self.state, -1, axis=1)
        state_bw = np.roll(self.state_bw, -1, axis=0)
        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state_bw[-1] = curr_bw
        if self.MLP:
            tput_pred = self.forecaster_model(torch.tensor(state_bw, dtype=torch.float32).to('cuda')).detach().cpu().numpy()
            netbw_pred = tput_pred
        else:
            tput_pred = self.forecaster_model(torch.tensor(state_bw, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to('cuda')).detach().cpu().numpy()
            netbw_pred = tput_pred.squeeze(0)
        tput_history = state_bw[-4:]
        state[4, :] = np.concatenate([tput_history, netbw_pred])
        state[5, :A_DIM] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[6, -1] = np.minimum(video_chunk_remain,
                                  CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        self.state = state
        self.state_bw = state_bw
        truncated = end_of_video
        #observation, reward, done, info = env.step(action)
        return state, reward, end_of_video, truncated, {'bitrate': VIDEO_BIT_RATE[bit_rate], 'rebuffer': rebuf}
