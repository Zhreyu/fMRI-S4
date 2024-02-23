from src.models.sequence.ss.s4 import S4
from tqdm.auto import tqdm
from src.tasks.encoders import PositionalEncoder, Conv1DEncoder
from src.tasks.decoders import SequenceDecoder
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix


device = 'cuda:0'

# model.py
class Conv1DEncoder(nn.Module):
    def __init__(self, n_layers, d_input, d_model, kernel_size=3, stride=1, padding=1):
        super().__init__()
        layers = [nn.Conv1d(d_input, d_model, kernel_size, stride, padding), nn.ReLU()]
        for _ in range(1, n_layers):
            layers.append(nn.Conv1d(d_model, d_model, kernel_size, stride, padding))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class EEGModel(pl.LightningModule):
    def __init__(self, n_conv_layers=2, d_input=20, d_model=256, n_s4_layers=2, T=1000, channels=1):
        super().__init__()
        self.encoder = Conv1DEncoder(n_conv_layers, d_input, d_model)
        self.s4_layers = nn.ModuleList([
            S4(d_model=d_model, l_max=T, bidirectional=True, dropout=0.2, transposed=True, channels=channels)
            for _ in range(n_s4_layers)
        ])
        self.decoder = nn.Linear(d_model, d_input)  # Adjust as necessary

    def forward(self, x):
        #print(f"Initial shape: {x.shape}")  # Log the initial shape of x
        x = x.transpose(1, 2)
       # print(f"Shape after transpose: {x.shape}")  # Shape after the first transpose

        x = self.encoder(x)
        #print(f"Shape after encoder: {x.shape}")  # Shape after passing through the encoder

        for layer in self.s4_layers:
            x, _ = layer(x)  # Assuming your S4 layer implementation can handle this shape
            #print(f"Shape after S4 layer: {x.shape}")  # Shape after each S4 layer

        # Assuming decoder expects [batch_size, channels, sequence_length]
        x = self.decoder(x.transpose(-1, -2)).transpose(-1, -2)  # This double transpose seems redundant, consider reviewing
        #print(f"Shape before final transpose: {x.shape}")  # Shape before the final transpose (if you keep the double transpose)
        x = x.permute(0, 2, 1)
        #print(f"Final output shape: {x.shape}")  # Final shape of x before returning

        return x


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        # Assuming batch is a tuple of (input_sequence, target_sequence)
        input_sequence, target_sequence = batch
        predicted_sequence = self(input_sequence)
        loss = F.mse_loss(predicted_sequence, target_sequence)
        self.log('train_loss', loss)
        return loss

