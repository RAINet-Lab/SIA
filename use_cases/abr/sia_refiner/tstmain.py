from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import TensorDataset, random_split, DataLoader
import matplotlib.pyplot as plt
import load_trace
from Patch_TST import PatchTST

TRACES = str(Path(__file__).resolve().parent / "test") + "/"
all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRACES)

bw_data = []
for i in range(len(all_cooked_bw)):
    for j in range(len(all_cooked_bw[i])):
        bw_data.append(all_cooked_bw[i][j])

zeros_to_add = [0] * 11
data = zeros_to_add + bw_data

def prepare_data(data, input_window=12, output_window=4):
    X, y = [], []
    for i in range(len(data) - input_window - output_window + 1):
        X.append(data[i:i+input_window])
        y.append(data[i+input_window:i+input_window+output_window])
    return np.array(X), np.array(y)


X, y = prepare_data(data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)

dataset = TensorDataset(X_tensor, y_tensor)
test_dataset = DataLoader(dataset, batch_size=32, shuffle=False)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataset = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=32, shuffle=False)


class EarlyStopping:
    def __init__(self, patience=100, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def step(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

class LRScheduler:
    def __init__(self, optimizer, patience=50, min_lr=1e-10, factor=0.5, verbose=False):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr, verbose=verbose)

    def step(self, loss):
        self.scheduler.step(loss)


# Define model
n_token = 12           # lookback window size
input_dim = 1         # univariate input
model_dim = 64        # model/embedding dimension
num_heads = 4         # number of heads for multi-head attention
num_layers = 3        # number of transformer layers
output_dim = 4        # output dimension = horizon

model = PatchTST(n_token=n_token, 
                 input_dim=input_dim, 
                 model_dim=model_dim, 
                 num_heads=num_heads, 
                 num_layers=num_layers, 
                 output_dim=output_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

early_stopping = EarlyStopping(patience=100, min_delta=1e-5)
scheduler = LRScheduler(optimizer=optimizer, patience=50, min_lr=1e-10, factor=0.5, verbose=True)


# Define training and evaluation function
def train_and_evaluate_model(model, train_loader, test_loader, epochs=2500, eval_interval=100, save_interval=200):
    test_loss_data = []
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)  # Forward pass
            y_batch = y_batch.squeeze(-1)  # Squeeze the last dimension to match output shape
            loss = criterion(output, y_batch)  # Compute loss
            running_train_loss += loss.item()
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize

        avg_train_loss = running_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

        # Evaluate the model
        if (epoch + 1) % eval_interval == 0:
            model.eval()
            running_test_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    output = model(X_batch)  # Forward pass
                    y_batch = y_batch.squeeze(-1)  # Squeeze the last dimension to match output shape
                    loss = criterion(output, y_batch)  # Compute loss
                    running_test_loss += loss.item()

            avg_test_loss = running_test_loss / len(test_loader)
            print(f"Test Loss at Epoch {epoch + 1}: {avg_test_loss:.4f}")

            # Early stopping and learning rate scheduler
            scheduler.step(avg_test_loss)
            if early_stopping.step(avg_test_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Save model
        if (epoch + 1) % save_interval == 0:
            save_path = str((Path(__file__).resolve().parent / "TS_models" / f"tst3_epoch_{epoch+1}_loss_{avg_test_loss:.4f}.pt"))
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1}")

    return test_loss_data


# Train and evaluate the model
test_loss = train_and_evaluate_model(
    model, train_dataset, test_dataset, epochs=1500, eval_interval=25, save_interval=25)