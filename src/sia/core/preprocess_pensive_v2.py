import numpy as np

# VIDEO_BIT_RATE = [300., 750., 1200., 1850., 2850., 4300.]
VIDEO_BIT_RATE = [20000, 40000, 60000, 80000, 110000, 160000]  # Kbps


def preprocess(data_dict):
    # 1. Keep only the last value [-1] in 'Download Chunk Throughput' and 'Download Chunk Delay'
    # data_dict['Download Chunk Throughput (Kbps/ms)'] = round(data_dict['Download Chunk Throughput (Kbps/ms)'], 3)
    # data_dict['Download Chunk Delay (Norm by 1/10 sec)'] = round(data_dict['Download Chunk Delay (Norm by 1/10 sec)'], 3)

    # 2. Remove 'Next Chunk Sizes (Mb)' key
    if 'Next Chunk Sizes (Mb)' in data_dict:
        del data_dict['Next Chunk Sizes (Mb)']

    # 3. Rename keys to match the desired column names
    renamed_data_dict = {
        'Timestep': data_dict.get('Timestep'),
        'File_name': data_dict.get('File Name'),
        'last_brate': data_dict.get('Prev Bitrate Ratio'),
        'buffer': data_dict.get('Buffer Size (Norm by 1/10 sec)'),
        'dl_tput': data_dict.get('Download Chunk Throughput (Kbps/ms)'),
        'dl_delay': data_dict.get('Download Chunk Delay (Norm by 1/10 sec)'),
        'rem_chunks': data_dict.get('Chunks Remain Ratio'),
        'sel_brate': data_dict.get('Selected Bitrate (Kbps)'),
        'reward': data_dict.get('reward')
    }

    # 4. Round all numeric values to 3 decimal places for the entire dictionary
    processed_data_dict = {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in renamed_data_dict.items()}

    return processed_data_dict

def AS_preprocess(input_state, agent_action):
    
    
    state = input_state.copy()
    
    # 1. Keep only the last value [-1] in 'Download Chunk Throughput' and 'Download Chunk Delay'
    # state['Download Chunk Throughput (Kbps/ms)'] = round(state['Download Chunk Throughput (Kbps/ms)'][-1], 3)
    # state['Download Chunk Delay (Norm by 1/10 sec)'] = round(state['Download Chunk Delay (Norm by 1/10 sec)'][-1], 3)

    # 2. Remove 'Next Chunk Sizes (Mb)' key
    if 'Next Chunk Sizes (Mb)' in state:
        del state['Next Chunk Sizes (Mb)']

    # 3. Rename keys to match the desired column names
    renamed_data_dict = {
        'Timestep': state.get('Timestep'),
        'File_name': state.get('File Name'),
        'last_brate': state.get('Prev Bitrate Ratio'),
        'buffer': state.get('Buffer Size (Norm by 1/10 sec)'),
        'dl_tput': state.get('Download Chunk Throughput (Kbps/ms)'),
        'dl_delay': state.get('Download Chunk Delay (Norm by 1/10 sec)'),
        'rem_chunks': state.get('Chunks Remain Ratio'),
        'sel_brate': VIDEO_BIT_RATE[agent_action],
        'reward': 0,
    }

    # 4. Round all numeric values to 3 decimal places for the entire dictionary
    processed_data_dict = {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in renamed_data_dict.items()}

    return processed_data_dict

def bitrate_to_acceptable_converter(as_bitrate):
    return VIDEO_BIT_RATE.index(as_bitrate)

def acceptable_to_bitrate_converter(as_bitrate):   
    return VIDEO_BIT_RATE.index(as_bitrate)

