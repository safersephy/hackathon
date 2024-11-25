from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


def load_from_parquet(file_path):
    """Load and reconstruct numpy arrays for WESAD dataset"""
    table = pq.read_table(file_path)

    # Convert back to numpy arrays - make them writable once during loading
    data = []
    for binary, shape0, shape1 in zip(
        table["data"].to_numpy(),
        table["shape_0"].to_numpy(),
        table["shape_1"].to_numpy(),
    ):
        # Make a writable array just once during loading
        arr = np.frombuffer(binary).reshape(shape0, shape1).copy().astype(np.float32)
        data.append(arr)

    labels = table["label"].to_numpy()
    subjects = table["subject"].to_numpy()

    return data, labels, subjects


class WESADDataset(Dataset):
    def __init__(self, parquet_path, window_size=1000):
        self.data, self.labels, self.subjects = load_from_parquet(parquet_path)
        self.window_size = window_size

        # Create index mapping for faster __getitem__
        self.index_map = []
        for chunk_idx, chunk_data in enumerate(self.data):
            n_windows = len(chunk_data) - window_size + 1
            if n_windows > 0:
                for window_idx in range(n_windows):
                    self.index_map.append((chunk_idx, window_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        chunk_idx, window_idx = self.index_map[idx]

        # Get the data chunk and its label
        chunk_data = self.data[chunk_idx]
        chunk_label = self.labels[chunk_idx]

        # Extract the window - no copy needed as arrays are already writable
        window_data = chunk_data[window_idx : window_idx + self.window_size]

        # Convert to torch tensor
        window_tensor = torch.from_numpy(window_data)
        label_tensor = torch.tensor(chunk_label, dtype=torch.long)

        return window_tensor, label_tensor


class SubsetDataset(Dataset):
    def __init__(self, dataset, indices, split_name="unknown"):
        self.dataset = dataset
        self.indices = indices
        self.split_name = split_name

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __str__(self):
        return f"WESAD {self.split_name} dataset with {len(self)} items"

    def __repr__(self):
        return f"WESAD {self.split_name} dataset with {len(self)} items"


class WESADTrainValidSplit:
    def __init__(self, dataset: WESADDataset, valid_ratio=0.2, seed=42):
        self.dataset = dataset
        np.random.seed(seed)

        # Get unique chunks
        unique_chunks = set(idx for idx, _ in dataset.index_map)
        n_chunks = len(unique_chunks)

        # Create mapping of chunk_idx to all its windows
        chunk_to_windows: dict = {}
        for i, (chunk_idx, window_idx) in enumerate(dataset.index_map):
            if chunk_idx not in chunk_to_windows:
                chunk_to_windows[chunk_idx] = []
            chunk_to_windows[chunk_idx].append(i)

        # Randomly split chunks
        n_valid_chunks = int(n_chunks * valid_ratio)
        valid_chunks = set(
            np.random.choice(list(unique_chunks), n_valid_chunks, replace=False)
        )

        # Create train and valid indices
        self.train_indices = []
        self.valid_indices = []

        for chunk_idx, window_indices in chunk_to_windows.items():
            if chunk_idx in valid_chunks:
                self.valid_indices.extend(window_indices)
            else:
                self.train_indices.extend(window_indices)

        # Create train and valid datasets
        self.train_dataset = SubsetDataset(dataset, self.train_indices, "train")
        self.valid_dataset = SubsetDataset(dataset, self.valid_indices, "validation")


class HeartDataset2D:
    def __init__(
        self,
        path: Path,
        target: str,
        shape: tuple[int, int] = (16, 12),
    ) -> None:
        self.df = pd.read_parquet(path)
        self.target = target
        _x = self.df.drop("target", axis=1)
        x = torch.tensor(_x.values, dtype=torch.float32)

        # original length is 187, which only allows for 11x17 2D tensors
        # 3*2**6 = 192. This makes it easier to reshape the data
        # it also makes convolutions / maxpooling more predictable
        self.x = torch.nn.functional.pad(x, (0, 3 * 2**6 - x.size(1))).reshape(
            -1, 1, *shape
        )
        y = self.df["target"]
        self.y = torch.tensor(y.values, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __repr__(self) -> str:
        return f"Heartdataset2D (#{len(self)})"


class HeartDataset1D:
    def __init__(
        self,
        path: Path,
        target: str,
    ) -> None:
        self.df = pd.read_parquet(path)
        self.target = target
        _x = self.df.drop("target", axis=1)
        x = torch.tensor(_x.values, dtype=torch.float32)
        # padded to 3*2**6 = 192
        # again, this helps with reshaping for attention & using heads
        self.x = torch.nn.functional.pad(x, (0, 3 * 2**6 - x.size(1)))
        y = self.df["target"]
        self.y = torch.tensor(y.values, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        # (seq_len, channels)
        return self.x[idx].unsqueeze(1), self.y[idx]

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __repr__(self) -> str:
        return f"Heartdataset (len {len(self)})"
