"""
Representation of the MIT-BIH ECG Arrhythmia Dataset.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import BaseDataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from collections import Counter


class ProteinDataset(BaseDataset):
    """
    Parameters
    ----------
    data_dir : str
        Path to directory containing 'mitbih_{train/test}.csv' files
    load_data : bool, optional
        Whether to load data on __init__, or delay until `load_data` call.
    normalize : bool, optional
        Whether to shift the data from range [0, 1] -> range[-1, 1]
    """

    def __init__(self, data_dir, load_data=True, normalize=None):
        self.data_dir = Path(data_dir)
        self.seq_path = self.data_dir / "XY.pt"
        assert self.seq_path.is_file(), "File must exist"

        self.sequence_length = 512
        self.input_shape = (self.sequence_length, 1)

        self.num_classes = 20
        self.output_shape = (self.num_classes,)

        if load_data:
            self.load_data()

    def data_dirname(self):
        return self.data_dir

    def load_data(self, normalize=None):
        """
        Define X/y train/test.
        """

        X, y = torch.load(self.seq_path)
        # X = X[:, np.newaxis]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )

    def __repr__(self):
        return (
            "Protein Family Classification Dataset\n"
            f"Num classes: {self.num_classes}\n"
            f"Input shape: {self.input_shape}\n"
            f"Train, Test counts: {self.X_train.shape[0]}, {self.X_test.shape[0]}\n"
        )
