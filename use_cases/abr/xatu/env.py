from pathlib import Path
# add queuing delay into halo
import os
import numpy as np
import core as abrenv
import fixed_env
import load_trace
import gymnasium as gym
from forecastor import Xatu, preprocess_data_from_dicts
import torch

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 7  # bit_rate, buffer_size, download throughput, download delay time, predicted download delay time, next_chunk_sizes, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = np.array([20000, 40000, 60000, 80000, 110000, 160000])  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 157.0
M_IN_K = 1000.0
REBUF_PENALTY = 160  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42

TEST_TRACES = str(Path(__file__).resolve().parent / 'test') + '/'
# TEST_TRACES = str(Path(__file__).resolve().parent / 'complete') + '/'

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_static_features = 2
num_temporal_features = 10
lstm_hidden_size = 128
mlp_hidden_size = 128
forecast_horizon = 1
SEQ_LENGTH = 5
PREDICTION_LENGTH = 1

model = Xatu(num_static_features, num_temporal_features, lstm_hidden_size, mlp_hidden_size, forecast_horizon)
model.load_state_dict(torch.load(Path(__file__).resolve().parent / 'For_models' / 'best_model_epoch_870_mape_14.64.pth'))
model.to(device)
model.eval()



class ABREnv(gym.Env):

    def __init__(self, random_seed=RANDOM_SEED, train=True, TEST_TRACES=TEST_TRACES):
        np.random.seed(random_seed)
        
        if train:
            all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace()
            self.all_file_names = all_file_names
            self.net_env = abrenv.Environment(all_cooked_time=all_cooked_time,
                                              all_cooked_bw=all_cooked_bw)
        else:
            all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)
            self.all_file_names = all_file_names
            self.net_env = fixed_env.Environment(all_cooked_time=all_cooked_time,
                                          all_cooked_bw=all_cooked_bw,
                                          random_seed=random_seed)

        self.last_bit_rate = DEFAULT_QUALITY
        self.buffer_size = 0.
        self.state = np.zeros((S_INFO, S_LEN))
        self.time_stamp = 0
        self.action_space = gym.spaces.Discrete(A_DIM)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(S_INFO, S_LEN), dtype=np.float32)
        self.forecastor = model
        # self.video_count = 0
        self.data_sequence = np.zeros((SEQ_LENGTH, num_static_features + num_temporal_features))
        
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
            end_of_video, video_chunk_remain, curr_bw, video_trace_id = \
            self.net_env.get_video_chunk(bit_rate)
        state = np.roll(self.state, -1, axis=1)

        delay = float(delay) - abrenv.LINK_RTT

        # if end_of_video:
        #     self.video_count += 1
        #     if self.video_count >= len(self.all_file_names):
        #         self.video_count = 0

        static_data = {
            'Connection Type': self.all_file_names[video_trace_id], 
        }

        temporal_data = {
            'Current Thrroughput': float(video_chunk_size) / \
            float(delay) / M_IN_K / 100., # kilo byte / ms,  # s_batch[:, 2, -1]
            'Network Bandwidth': curr_bw,  # info['bandwidth']
            'Current Delivery Time': float(delay) / M_IN_K / BUFFER_NORM_FACTOR,  # s_batch[:, 3, :]
            'Previous Bitrate': VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE)),  # s_batch[:, 0, -1]
            'Next Chunk Sizes': (np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K / 100.).tolist(),  # s_batch[:, 5, :6]
        }

        data_array = preprocess_data_from_dicts(static_data, temporal_data)
        self.data_sequence = np.roll(self.data_sequence, shift=-1, axis=0)
        self.data_sequence[-1, :] = data_array

        static_features = torch.FloatTensor(self.data_sequence[:, :2]).to(device)  # First 2 features are static
        temporal_features = torch.FloatTensor(self.data_sequence[:, 2:]).to(device)  # Next 10 features are temporal
        predicted_delay = self.forecastor(static_features.unsqueeze(0), temporal_features.unsqueeze(0))

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K / 100. # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, -1] = predicted_delay.detach().cpu().numpy() # predicted download delay time
        state[5, :A_DIM] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K / 100.  # mega byte
        state[6, -1] = np.minimum(video_chunk_remain,
                                  CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        self.state = state
        return state, {'bitrate': VIDEO_BIT_RATE[bit_rate], 'rebuffer': rebuf, 'bandwidth': curr_bw}

    def render(self):
        return

    def step(self, action):
        bit_rate = int(action)
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, curr_bw, video_trace_id = \
            self.net_env.get_video_chunk(bit_rate)
        
        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smooth penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                      VIDEO_BIT_RATE[self.last_bit_rate]) / M_IN_K
        reward /= 100.
        self.last_bit_rate = bit_rate
        state = np.roll(self.state, -1, axis=1)

        delay = float(delay) - abrenv.LINK_RTT

        # if end_of_video:
        #     self.video_count += 1
        #     if self.video_count >= len(self.all_file_names):
        #         self.video_count = 0

        
        # print(self.video_count)

        static_data = {
            'Connection Type': self.all_file_names[video_trace_id], 
        }

        # print(static_data)

        temporal_data = {
            'Current Thrroughput': float(video_chunk_size) / \
            float(delay) / M_IN_K / 100., # kilo byte / ms,  # s_batch[:, 2, -1]
            'Network Bandwidth': curr_bw,  # info['bandwidth']
            'Current Delivery Time': float(delay) / M_IN_K / BUFFER_NORM_FACTOR,  # s_batch[:, 3, :]
            'Previous Bitrate': VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE)),  # s_batch[:, 0, -1]
            'Next Chunk Sizes': (np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K / 100.).tolist(),  # s_batch[:, 5, :6]
        }

        data_array = preprocess_data_from_dicts(static_data, temporal_data)
        self.data_sequence = np.roll(self.data_sequence, shift=-1, axis=0)
        self.data_sequence[-1, :] = data_array

        static_features = torch.FloatTensor(self.data_sequence[:, :2]).to(device)  # First 2 features are static
        temporal_features = torch.FloatTensor(self.data_sequence[:, 2:]).to(device)  # Next 10 features are temporal
        predicted_delay = self.forecastor(static_features.unsqueeze(0), temporal_features.unsqueeze(0))

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K / 100  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, -1] = predicted_delay.detach().cpu().numpy() # predicted download delay time
        state[5, :A_DIM] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K / 100  # mega byte
        state[6, -1] = np.minimum(video_chunk_remain,
                                  CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        self.state = state
        truncated = end_of_video
        #observation, reward, done, info = env.step(action)
        return state, reward, end_of_video, truncated, {'bitrate': VIDEO_BIT_RATE[bit_rate], 'rebuffer': rebuf, 'bandwidth': curr_bw}
