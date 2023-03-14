"""
Representation of the MIT-BIH ECG Arrhythmia Dataset.
"""
from pathlib import Path

import numpy as np
import pandas as pd

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
        self.seq_path = self.data_dir / "pdb_data_seq.csv"
        assert self.seq_path.is_file(), "File must exist"

        self.nodup_path = self.data_dir / "pdb_data_no_dups.csv"
        assert self.nodup_path.is_file(), "File must exist"

        self.sequence_length = 1200
        self.input_shape = (self.sequence_length, 1)

        self.num_classes = 10
        self.output_shape = (self.num_classes,)

        if load_data:
            self.load_data()

    def data_dirname(self):
        return self.data_dir

    def load_data(self, normalize=None):
        """
        Define X/y train/test.
        """

        df = (
            pd.read_csv(self.seq_path)
            .merge(pd.read_csv(self.nodup_path), how="inner", on="structureId")
            .drop_duplicates(["sequence"])
        )

        # Drop rows with missing labels
        df = df[[type(c) == type("") for c in df.classification.values]]
        df = df[[type(c) == type("") for c in df.sequence.values]]
        # select proteins
        df = df[df.macromoleculeType_x == "Protein"]
        df.reset_index()

        df = df.loc[df.residueCount_x < 1200]

        # count numbers of instances per class
        cnt = Counter(df.classification)
        # select only K most common classes! - was 10 by default
        top_classes = 10
        # sort classes
        sorted_classes = cnt.most_common()[:top_classes]
        classes = [c[0] for c in sorted_classes]
        counts = [c[1] for c in sorted_classes]
        print("at least " + str(counts[-1]) + " instances per class")

        # apply to dataframe
        print(str(df.shape[0]) + " instances before")
        df = df[[c in classes for c in df.classification]]
        print(str(df.shape[0]) + " instances after")

        seqs = df.sequence.values
        lengths = [len(s) for s in seqs]

        # Transform labels to one-hot
        lb = LabelEncoder()
        Y = lb.fit_transform(df.classification.tolist())
        self.classes = lb.classes_

        vocab = ["<pad>"]

        X = np.zeros((len(seqs), self.sequence_length)).astype(int)
        for i, (s, l) in enumerate(zip(seqs, lengths)):
            t = []
            for c in s:
                if c not in vocab:
                    vocab.append(c)
                t.append(vocab.index(c))
            X[i, :l] = t

        self.vocab = vocab

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, Y, test_size=0.2
        )

        self.class_weights = (
            1 - (np.bincount(self.y_train) / self.y_train.shape[0])
        ).tolist()

    def __repr__(self):
        return (
            "Protein Family Classification Dataset\n"
            f"Num classes: {self.num_classes}\n"
            f"Input shape: {self.input_shape}\n"
            f"Train, Test counts: {self.X_train.shape[0]}, {self.X_test.shape[0]}\n"
            f"Class weights: {self.class_weights}\n"
        )
