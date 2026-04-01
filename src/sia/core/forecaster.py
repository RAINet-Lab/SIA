import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import torch
import torch.nn as nn

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")


############### Define Forecaster ################
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, 
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.network = nn.Sequential(
            TemporalBlock(num_inputs, 10, kernel_size=5, stride=1, dilation=1, 
                          padding=4, dropout=dropout),
            TemporalBlock(10, 10, kernel_size=5, stride=1, dilation=2, 
                          padding=8, dropout=dropout)
        )

    def forward(self, x):
        return self.network(x)

class ST_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(ST_LSTM, self).__init__()
        self.tcn = TemporalConvNet(input_size, dropout=dropout)
        self.lstm = nn.LSTM(10, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.relu(out)  # ReLU after fully connected layer
        out = self.output(out)
        return out


class Forecaster:
    def __init__(self, base_dir, kpi_configs):
        """
        Initialize the forecaster with models for different KPIs.
        
        Args:
            base_dir (str): Base directory containing model files
            kpi_configs (dict): Dictionary mapping KPIs to their run IDs
        """
        self.base_dir = base_dir
        self.models = {}
        self.configs = {}
        self.preprocessing_infos = {}
        self.scalers = {}
        self.features = {}
        
        # Load models for each KPI
        for target_kpi, run_id in kpi_configs.items():
            # Define features for this KPI
            self.features[target_kpi] = [
                'num_ues', 'slice_prb', 'scheduling_policy', 
                'sum_requested_prbs', 'sum_granted_prbs', 
                'dl_n_samples', 'dl_mcs', 'dl_cqi_mean', 
                'dl_cqi_min', target_kpi
            ]
            
            # Load model and configurations
            model, model_config, preprocessing_info, scalers = self._load_model_and_config(
                target_kpi, run_id
            )
            
            self.models[target_kpi] = model
            self.configs[target_kpi] = model_config
            self.preprocessing_infos[target_kpi] = preprocessing_info
            self.scalers[target_kpi] = scalers

    def _load_model_and_config(self, target_kpi, run_id):
        """Load model and its configuration for a specific KPI."""
        run_dir = os.path.join(self.base_dir, target_kpi, run_id)
        
        # Load model configuration
        with open(os.path.join(run_dir, "model_config.json"), 'r') as f:
            model_config = json.load(f)
        
        # Load preprocessing information
        with open(os.path.join(run_dir, "preprocessing_info.json"), 'r') as f:
            preprocessing_info = json.load(f)
        
        
        # Load the best model
        training_results_dir = os.path.join(run_dir, "training_results")
        model_files = [f for f in os.listdir(training_results_dir) if f.endswith('.pth')]
        best_model_file = sorted(model_files, key=lambda x: float(x.split('_')[1].split('-')[1]))[0]
        model = torch.load(os.path.join(training_results_dir, best_model_file))
        
        # Load scalers if they exist
        scalers = None
        scaler_path = os.path.join(run_dir, "scalers.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
        
        return model, model_config, preprocessing_info, scalers

    def _apply_sg_filter(self, df, features, window_length=5, polyorder=2):
        """Apply Savitzky-Golay filter to specified features."""
        for feature in features:
            df[feature] = savgol_filter(df[feature], window_length, polyorder)
        return df

    def _scale_features_deploy(self, df, features, scalers):
        """Scale features using provided scalers."""
        for feature in features:
            # scaler = StandardScaler()
            df[feature] = scalers[feature].transform(df[[feature]])
            # df[feature] = scaler.fit_transform(df[[feature]])
            
        return df, scalers

    def _apply_log_transform(self, df, features):
        """Apply log transform to specified features."""
        for feature in features:
            df[feature] = np.log1p(df[feature])  # log1p is log(1+x), which handles zero values
        return df

    def _preprocess_data(self, data, target_kpi):
        """Preprocess input data according to the KPI's preprocessing configuration."""
        features = self.features[target_kpi]
        preprocessing_info = self.preprocessing_infos[target_kpi]
        scalers = self.scalers[target_kpi]
        
        preprocessed_data = data.copy()
        
        for step in preprocessing_info['steps']:
            if step == 'log':
                preprocessed_data = self._apply_log_transform(preprocessed_data, features)
            elif step == 'denoise':
                preprocessed_data = self._apply_sg_filter(preprocessed_data, features)
            elif step == 'scaler' and scalers is not None:
                preprocessed_data, _ = self._scale_features_deploy(preprocessed_data, features, scalers)
        
        return preprocessed_data

    def _postprocess_predictions(self, predictions, target_kpi):
        """Reverse preprocessing steps for predictions."""
        preprocessing_info = self.preprocessing_infos[target_kpi]
        scalers = self.scalers[target_kpi]
        
        reverted_predictions = predictions.copy()
        
        for step in reversed(preprocessing_info['steps']):
            if step == 'scaler' and scalers is not None:
                scaler = scalers[target_kpi]
                reverted_predictions = scaler.inverse_transform(
                    reverted_predictions.reshape(-1, 1)
                ).flatten().reshape(reverted_predictions.shape)
            elif step == 'log':
                reverted_predictions = np.expm1(reverted_predictions)
        
        return reverted_predictions

    def _create_dataset(self, df, target_kpi, window_size=10):
        """
        Create model input dataset from preprocessed data.
        Assumes the input df has exactly window_size timesteps.
        """
        features = self.features[target_kpi]
        
        # Just take all timesteps as one sample since we're getting exactly window_size timesteps
        X = df[features].values.reshape(1, window_size, -1)
        return torch.FloatTensor(X).to(device)

    def forecast(self, data, target_kpi):
        """
        Generate forecasts for the given data and KPI.
        Expects exactly 10 timesteps of data.
        
        Args:
            data (pd.DataFrame): Input data containing required features (exactly 10 timesteps)
            target_kpi (str): The KPI to forecast
            
        Returns:
            np.array: Array of predictions
        """
        if len(data) != 10:
            raise ValueError(f"Input data must contain exactly 10 timesteps. Got {len(data)} timesteps.")

        # Preprocess the input data
        preprocessed_data = self._preprocess_data(data, target_kpi)
        
        # Create model input dataset
        X = self._create_dataset(preprocessed_data, target_kpi)
        
        # Generate predictions
        model = self.models[target_kpi]
        model.eval()
        with torch.no_grad():
            predictions = model(X)
            predictions = predictions.cpu().numpy()
        
        # Postprocess predictions
        reverted_predictions = self._postprocess_predictions(predictions, target_kpi)
        
        return reverted_predictions
