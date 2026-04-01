import pandas as pd
from datetime import datetime

def compute_autoencoder_values(autoencoder_data, mode):
    if mode == "mean":
        return autoencoder_data.mean().to_dict()
    elif mode == "median":
        return autoencoder_data.median().to_dict()
    elif mode == "sum":  # New mode added for summing the items
        return autoencoder_data.sum().to_dict()
    elif mode == "clean_mean":
        filtered_data = autoencoder_data[(autoencoder_data != 0).any(axis=1)]
        return filtered_data.mean().to_dict()
    elif mode == "clean_median":
        filtered_data = autoencoder_data[(autoencoder_data != 0).any(axis=1)]
        return filtered_data.median().to_dict()
    elif mode == "adv_median":
        percentiles = autoencoder_data.quantile([0.25, 0.5, 0.75])
        return (0.5 * percentiles.loc[0.5] + 0.25 * percentiles.loc[0.25] + 0.25 * percentiles.loc[0.75]).to_dict()
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def create_dataframe(data_dict, mode="mean"):
    # Extracting data from the input dictionary
    autoencoder_input = data_dict['AUTOENCODER_INPUT']
    prb_values = data_dict['PRB']
    scheduling_values = data_dict['SCHEDULING']
    timestamp = data_dict.get('TIMESTAMP')
    
    # Convert timestamp to datetime format if present
    timestamp_dt = pd.to_datetime(timestamp, unit='ms')

    # Preparing data for the DataFrame
    rows = []
    for slice_id in range(2):
        autoencoder_data = autoencoder_input[1 - slice_id]  # Notice: input is flipped for slice 1 and slice 0
        prb = prb_values[slice_id]
        scheduling_policy = scheduling_values[slice_id]

        autoencoder_values = compute_autoencoder_values(autoencoder_data, mode)

        row = {
            'timestamp': timestamp_dt,
            'slice_id': slice_id,
            'slice_prb': prb,
            'scheduling_policy': scheduling_policy,
            'tx_brate downlink [Mbps]': autoencoder_values['tx_brate downlink [Mbps]'],
            'tx_pkts downlink': autoencoder_values['tx_pkts downlink'],
            'dl_buffer [bytes]': autoencoder_values['dl_buffer [bytes]'],
        }
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    return df
