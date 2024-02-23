import pandas as pd
import numpy as np
from torchvision.transforms import Compose
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler
from tsaug import Resize


DATASETS = ['Mddrest','Abide', 'Ukbb', 'Jpmdd', 'Synth','EEG']

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def resize(x_, t):
            x_ = np.expand_dims(x_,0)
            x_ = Resize(size=t).augment(x_)
            x_ = np.squeeze(x_,0)
            return x_

class Abstract_Dataset(Dataset):
    """
    An Abstract class for all Datasets



    """
    def __init__(self):
        return


    def __read_data__(self):

        return

    def __getitem__(self, index):

                return torch.tensor(self.tc_data[index].astype(np.float32)), torch.tensor(self.labels[index].astype(np.int64))

    def __len__(self):
        return self.total_subjects



import os
import numpy as np
import torch
from torch.utils.data import Dataset
import mne

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import mne

class EEGDataset(Dataset):
    """
    A PyTorch Dataset class to load EEG data for self-supervised learning
    """
    def __init__(self, data_folder, sequence_length=1000, transform=None):
        """
        Args:
            data_folder (string): Path to the folder containing .fif files.
            sequence_length (int): Length of the sequences for the model to predict the next timestep.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_folder = data_folder
        self.sequence_length = sequence_length
        self.transform = transform
        self.filenames = [f for f in os.listdir(data_folder) if f.endswith('.fif')]

        # Here you load the actual data. Depending on dataset size, you might need to load per item
        # or implement a more sophisticated caching strategy.
        self.data = []
        for filename in self.filenames:
            file_path = os.path.join(self.data_folder, filename)
            eeg_data = self.load_eeg_data(file_path)
            self.data.append(eeg_data)

    def load_eeg_data(self, file_path):
        """
        Load preprocessed EEG data from a .fif file.
        """
        eeg_data = mne.io.read_raw_fif(file_path, preload=True, verbose='ERROR')
        data = eeg_data.get_data().transpose()  # Shape: (timepoints, channels)
        return data

    def __len__(self):
        # We need to account for the fact that each file can produce multiple sequences
        return sum(max(0, data.shape[0] - self.sequence_length) for data in self.data)

    def __getitem__(self, idx):
        # Find which data file the index corresponds to
        file_idx, seq_start = self.index_to_sequence_position(idx)

        # Retrieve the sequence and the next timestep
        sequence = self.data[file_idx][seq_start:seq_start + self.sequence_length]
        target = self.data[file_idx][seq_start + 1:seq_start + self.sequence_length + 1]

        if self.transform:
            sequence = self.transform(sequence)
            target = self.transform(target)

        sequence_tensor = torch.from_numpy(sequence).float()
        target_tensor = torch.from_numpy(target).float()

        return sequence_tensor, target_tensor

    def index_to_sequence_position(self, idx):
        # Convert a flat index to the corresponding file index and sequence start position
        for file_idx, data in enumerate(self.data):
            if idx < max(0, data.shape[0] - self.sequence_length):
                return file_idx, idx
            idx -= max(0, data.shape[0] - self.sequence_length)
        raise IndexError("Index out of bounds")

# Example usage:
dataset = EEGDataset(data_folder='/content/drive/MyDrive/01_tcp_ar', sequence_length=1000)

