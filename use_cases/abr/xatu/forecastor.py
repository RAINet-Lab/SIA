import torch
import torch.nn as nn
import numpy as np

class Xatu(nn.Module):
    def __init__(self, num_static_features, num_temporal_features, lstm_hidden_size, mlp_hidden_size, forecast_horizon):
        super(Xatu, self).__init__()

        self.forecast_horizon = forecast_horizon

        # Static Feature Embedding
        self.static_embed = nn.ModuleList([nn.Linear(1, mlp_hidden_size) for _ in range(num_static_features)])
        self.static_fc = nn.Linear(num_static_features * mlp_hidden_size, lstm_hidden_size)
        self.gate_mask = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size),
            nn.Sigmoid()
        )

        # Temporal Feature Processing
        self.lstm = nn.LSTM(num_temporal_features, lstm_hidden_size, batch_first=True)
        self.temporal_fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size),
            nn.ReLU()
        )

        # Prediction Layer
        self.predict_fc = nn.Linear(lstm_hidden_size, forecast_horizon)

    def forward(self, static_features, temporal_features):
        # Static Feature Embedding
        static_embeddings = [self.static_embed[i](static_features[:,:,i:i+1]) for i in range(static_features.shape[2])]
        static_embeddings = torch.cat(static_embeddings, dim=2)
        static_embeddings = self.static_fc(static_embeddings.view(static_embeddings.size(0), static_embeddings.size(1), -1))
        gate = self.gate_mask(static_embeddings)

        # Temporal Feature Processing
        lstm_out, _ = self.lstm(temporal_features)
        temporal_embeddings = self.temporal_fc(lstm_out)

        # Combining static and temporal features with gate mask
        masked_embeddings = gate * temporal_embeddings

        # Prediction
        prediction = self.predict_fc(masked_embeddings[:, -1, :])  # Use only the last time step
        return prediction.unsqueeze(-1)  # Add an extra dimension to match (batch_size, forecast_horizon, 1)
    

def preprocess_data_from_dicts(static_data, temporal_data):
    # Function to categorize connection type and trace type
    def categorize_connection_type(connection_type):
        if '5g' in connection_type:
            return 1
        elif '4g' in connection_type:
            return 0
        else:
            return -1  # Unknown category

    def categorize_trace_type(connection_type):
        if 'driving' in connection_type:
            return 1
        elif 'walking' in connection_type:
            return 0
        else:
            return -1  # Unknown category

    # Process static data
    connection_type = static_data['Connection Type']
    connection_type_category = categorize_connection_type(connection_type)
    trace_type_category = categorize_trace_type(connection_type)

    # Process temporal data
    current_throughput = temporal_data['Current Thrroughput']
    network_bandwidth = temporal_data['Network Bandwidth']
    current_delivery_time = temporal_data['Current Delivery Time']
    previous_bitrate = temporal_data['Previous Bitrate']
    next_chunk_sizes = temporal_data['Next Chunk Sizes']

    # Ensure next_chunk_sizes has exactly 6 elements (pad with zeros if necessary)
    next_chunk_sizes = next_chunk_sizes + [0] * (6 - len(next_chunk_sizes)) if len(next_chunk_sizes) < 6 else next_chunk_sizes[:6]

    # Combine all variables into a numpy array
    data_array = np.array([
        connection_type_category,
        trace_type_category,
        current_throughput,
        network_bandwidth,
        current_delivery_time,
        previous_bitrate,
        *next_chunk_sizes
    ])

    data_array = [int(data_array[0]), int(data_array[1])] + data_array[2:].tolist()

    return data_array

