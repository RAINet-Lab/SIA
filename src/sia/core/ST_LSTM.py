import torch
import torch.nn as nn

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

