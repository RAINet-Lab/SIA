import torch
import torch.nn as nn

n_token = 12           # lookback window size
input_dim = 1         # univariate input
model_dim = 64        # model/embedding dimension
num_heads = 4         # number of heads for multi-head attention
num_layers = 3        # number of transformer layers
output_dim = 4        # output dimension = horizon

class PatchTST(nn.Module):
    def __init__(self, n_token, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(PatchTST, self).__init__()
        self.patch_embedding = nn.Linear(input_dim, model_dim)    # Input Embedding
        self._pos = torch.nn.Parameter(torch.randn(1,1,model_dim))  # Positional Embedding

        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.output_layer = nn.Linear(model_dim * n_token, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        # x shape: (batch_size, n_token, token_size)
        x = self.patch_embedding(x)   # (batch_size, n_token, model_dim)
        x = x + self._pos             # Add positional embedding
        x = self.transformer_encoder(x)   # (batch_size, n_token, model_dim)
        x = x.view(x.size(0), -1)     # Flatten to (batch_size, n_token * model_dim)
        output = self.output_layer(x) # (batch_size, output_dim)
        output = self.relu(output)
        return output
    

# class ThroughputForecastNNold(nn.Module):
#     def __init__(self, input_size=12, output_size=4, hidden_size1=256, hidden_size2=128, hidden_size3=64, hidden_size4=32):
#         super(ThroughputForecastNNold, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size3)
#         self.fc2 = nn.Linear(hidden_size3, hidden_size3)
#         self.fc3 = nn.Linear(hidden_size3, hidden_size1)
#         self.fc4 = nn.Linear(hidden_size1, output_size)
#         # self.fc5 = nn.Linear(hidden_size1, output_size)
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = torch.relu(self.fc4(x)) 
#         # x = torch.relu(self.fc5(x)) 
#         return x