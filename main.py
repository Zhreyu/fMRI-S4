import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import EEGModel
from datasets import EEGDataset  # Adjust import path as needed

def train_model():
    # Set up dataset and dataloader
    dataset = EEGDataset(data_folder='/content/drive/MyDrive/01_tcp_ar', sequence_length=20)  # Adjust as needed
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8)

    # Initialize the model
    model = EEGModel(n_conv_layers=2, d_input=20, d_model=256, n_s4_layers=2, T=1000, channels=1)

    # Set up PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=5)  # Adjust as needed
    trainer.fit(model, dataloader)
    return model

if __name__ == "__main__":
      model = train_model()
