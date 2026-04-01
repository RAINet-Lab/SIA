import numpy as np
import pandas as pd

VIDEO_BIT_RATE = [300., 750., 1200., 1850., 2850., 4300.]
# VIDEO_BIT_RATE = [20000, 40000, 60000, 80000, 110000, 160000]  # Kbps

def preprocess(data_dict):
    # 1. Keep only the last value [-1] in 'Download Chunk Throughput' and 'Download Chunk Delay'
    data_dict['Download Chunk Throughput (Kbps/ms)'] = round(data_dict['Download Chunk Throughput (Kbps/ms)'][-1], 3)
    data_dict['Download Chunk Delay (Norm by 1/10 sec)'] = round(data_dict['Download Chunk Delay (Norm by 1/10 sec)'][-1], 3)
    # data_dict['Bandwidth (Kbps)'] = round(data_dict['Bandwidth (Kbps)'][-1], 3)
    # data_dict['Download Chunk Throughput (Kbps/ms)'] = round(pd.Series(data_dict['Download Chunk Throughput (Kbps/ms)']).mean(), 3)
    # data_dict['Download Chunk Delay (Norm by 1/10 sec)'] = round(pd.Series(data_dict['Download Chunk Delay (Norm by 1/10 sec)']).mean(), 3)
    # data_dict['Bandwidth (Kbps)'] = round(pd.Series(data_dict['Bandwidth (Kbps)'][-4:]).mean(), 3)

    # 2. Remove 'Next Chunk Sizes (Mb)' key if it exists
    data_dict.pop('Next Chunk Sizes (Mb)', None)

    # 3. Rename keys to match the desired column names
# <<<<<<< HEAD
#     renamed_keys = {
#         'Timestep': 'Timestep',
#         'File Name': 'File_name',
#         'Prev Bitrate Ratio': 'last_brate',
#         'Buffer Size (Norm by 1/10 sec)': 'buffer',
#         'Download Chunk Throughput (Kbps/ms)': 'dl_tput',
#         'Download Chunk Delay (Norm by 1/10 sec)': 'dl_delay',
#         'Chunks Remain Ratio': 'rem_chunks',
#         'Selected Bitrate (Kbps)': 'sel_brate',
#         'reward': 'reward'
# =======
    renamed_data_dict = {
        'Timestep': data_dict.get('Timestep'),
        'File_name': data_dict.get('File Name'),
        'last_brate': data_dict.get('Prev Bitrate Ratio'),
        'buffer': data_dict.get('Buffer Size (Norm by 1/10 sec)'),
        'dl_tput': data_dict.get('Download Chunk Throughput (Kbps/ms)'),
        'dl_delay': data_dict.get('Download Chunk Delay (Norm by 1/10 sec)'),
        # 'bwidth': data_dict.get('Bandwidth (Kbps)'),
        'rem_chunks': data_dict.get('Chunks Remain Ratio'),
        'sel_brate': data_dict.get('Selected Bitrate (Kbps)'),
        'reward': data_dict.get('reward')
    }
    
    processed_data_dict = {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in renamed_data_dict.items()}

    return processed_data_dict

def AS_preprocess(input_state, agent_action):
    """
    Preprocess the agent's input state for further processing.
    
    Args:
        input_state (dict): The input state dictionary.
        agent_action (int): The action taken by the agent.
    
<<<<<<< HEAD
    Returns:
        dict: The processed state dictionary.
    """
    # Create a copy of the input state to avoid modifying the original
    state = input_state.copy()
    # 1. Keep only the last value [-1] in 'Download Chunk Throughput' and 'Download Chunk Delay'
    state['Download Chunk Throughput (Kbps/ms)'] = round(state['Download Chunk Throughput (Kbps/ms)'][-1], 3)
    state['Download Chunk Delay (Norm by 1/10 sec)'] = round(state['Download Chunk Delay (Norm by 1/10 sec)'][-1], 3)
    # state['Download Chunk Throughput (Kbps/ms)'] = round(pd.Series(state['Download Chunk Throughput (Kbps/ms)']).mean(), 3)
    # state['Download Chunk Delay (Norm by 1/10 sec)'] = round(pd.Series(state['Download Chunk Delay (Norm by 1/10 sec)']).mean(), 3)

    # # 1. Round each value in 'Download Chunk Throughput' and 'Download Chunk Delay' to 3 decimal places
    # keys_to_round = [
    #     'Download Chunk Throughput (Kbps/ms)',
    #     'Download Chunk Delay (Norm by 1/10 sec)'
    # ]
    # for key in keys_to_round:
    #     state[key] = [round(x, 3) for x in state.get(key, [])]

    # 2. Remove 'Next Chunk Sizes (Mb)' key if it exists
    state.pop('Next Chunk Sizes (Mb)', None)

    # 3. Rename keys to match the desired column names and set 'sel_brate' based on agent_action
    renamed_data_dict = {
        'Timestep': state.get('Timestep'),
        'File_name': state.get('File Name'),
        'last_brate': state.get('Prev Bitrate Ratio'),
        'buffer': state.get('Buffer Size (Norm by 1/10 sec)'),
        'dl_tput': state.get('Download Chunk Throughput (Kbps/ms)'),
        'dl_delay': state.get('Download Chunk Delay (Norm by 1/10 sec)'),
        'rem_chunks': state.get('Chunks Remain Ratio'),
        'sel_brate': VIDEO_BIT_RATE[agent_action] if agent_action < len(VIDEO_BIT_RATE) else None,
        'reward': 0,
    }

    # 4. Round all numeric values to 3 decimal places
    processed_data_dict = {
        k: (round(v, 3) if isinstance(v, (int, float)) else v)
        for k, v in renamed_data_dict.items()
    }

    return processed_data_dict

def bitrate_to_acceptable_converter(as_bitrate):
    """
    Convert the bitrate to its corresponding index in VIDEO_BIT_RATE list.
    
    Args:
        as_bitrate (float): The bitrate value to convert.
    
    Returns:
        int: The index of the bitrate in VIDEO_BIT_RATE, -1 if not found.
    """
    return VIDEO_BIT_RATE.index(as_bitrate)

def acceptable_to_bitrate_converter(as_bitrate):
    """
    Convert the acceptable bitrate to its corresponding index in VIDEO_BIT_RATE list.
    
    Args:
        as_bitrate (float): The acceptable bitrate value to convert.
    
    Returns:
        int: The index of the acceptable bitrate in VIDEO_BIT_RATE, -1 if not found.
    """
    return VIDEO_BIT_RATE.index(as_bitrate)