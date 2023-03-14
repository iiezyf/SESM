"""
Representation of the MIT-BIH ECG Arrhythmia Dataset.
"""
from pathlib import Path

import numpy as np

from datasets import BaseDataset


class DiagnosticDataset(BaseDataset):
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

    def __init__(self, data_dir, load_data=True, normalize=True):
        self.data_dir = Path(data_dir)
        self.normal_path = self.data_dir / "ptbdb_normal.csv"
        assert self.normal_path.is_file(), "File must exist"

        self.abnormal_path = self.data_dir / "ptbdb_abnormal.csv"
        assert self.abnormal_path.is_file(), "File must exist"

        self.sequence_length = 187
        self.input_shape = (self.sequence_length, 1)

        self.num_classes = 2
        self.output_shape = (self.num_classes,)

        self.normalize = normalize
        if load_data:
            self.load_data(normalize=self.normalize)

    def data_dirname(self):
        return self.data_dir

    def load_data(self, normalize=None):
        """
        Define X/y train/test.
        """
        normalize = normalize if normalize is not None else self.normalize

        normal_data = np.loadtxt("data/ptbdb_normal.csv", delimiter=",")
        abnormal_data = np.loadtxt("data/ptbdb_abnormal.csv", delimiter=",")

        all_data = np.vstack([normal_data, abnormal_data])

        X, y = all_data[:, :-1, np.newaxis], all_data[:, -1].astype(int)
        indices = np.arange(len(y))
        X = X[indices]
        y = y[indices]

        self.X_train, self.X_test = np.split(X, [0.8 * len(indices)])
        self.y_train, self.y_test = np.split(y, [0.8 * len(indices)])

    def __repr__(self):
        return (
            "PTB Diagnostic ECG Database\n"
            f"Num classes: {self.num_classes}\n"
            f"Input shape: {self.input_shape}\n"
            f"Train, Test counts: {self.X_train.shape[0]}, {self.X_test.shape[0]}\n"
        )
