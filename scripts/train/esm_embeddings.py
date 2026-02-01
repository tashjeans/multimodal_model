# various imports
import esm
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import sys
import os
import time
from tqdm.notebook import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from IPython.display import display, update_display
from torch.cuda.amp import GradScaler, autocast
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


device = "cuda" if torch.cuda.is_available() else "cpu"

model = ESMC.from_pretrained("esmc_300m").to(device)

protein = ESMProtein(sequence="AACGTATTTA")
protein_tensor = model.encode(protein)
logits_output = model.logits(
   protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
)
print(logits_output.logits)
print( logits_output.embeddings.size())

# class TCR_alpha_dataset(Dataset):
# class TCR_beta_dataset(Dataset):
# class peptide_dataset(Dataset):
# class HLA_dataset(Dataset):