import numpy as np

def preprocess_data_from_dicts(data):
    # Function to categorize connection type
    def categorize_connection_type(connection_type):
        if '5g' in connection_type and 'driving' in connection_type:
            return 0
        elif '5g' in connection_type and 'walking' in connection_type:
            return 1
        elif '4g' in connection_type and 'driving' in connection_type:
            return 2
        elif '4g' in connection_type and 'walking' in connection_type:
            return 3
        else:
            return -1  # Unknown category

    # Process  data
    connection_type = data['Connection Type']
    connection_type_category = categorize_connection_type(connection_type)
    Max_Historical_Throughput = data['Max Historical Throughput']
    current_throughput = data['Current Thrroughput']
    Max_Historical_Delivery_Time = data['Max Historical Delivery Time']
    Chunk_Index = data['Chunk Index']
    Players_State = 0 if data['Players State'] <= 0 else 1
    previous_bitrate = data['Previous Bitrate']
    next_chunk_sizes = data['Next Chunk Sizes']

    # Ensure next_chunk_sizes has exactly 6 elements (pad with zeros if necessary)
    next_chunk_sizes = next_chunk_sizes + [0] * (6 - len(next_chunk_sizes)) if len(next_chunk_sizes) < 6 else next_chunk_sizes[:6]

    # Combine all variables into a numpy array
    data_array = np.array([
        connection_type_category,
        Max_Historical_Throughput,
        current_throughput,
        Max_Historical_Delivery_Time,
        Chunk_Index,
        Players_State,
        previous_bitrate,
        *next_chunk_sizes
    ])

    data_array = [int(data_array[0]), int(data_array[1])] + data_array[2:].tolist()

    return data_array