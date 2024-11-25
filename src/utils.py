from mads_datasets.base import BaseDatastreamer
from mltrainer.preprocessors import BasePreprocessor
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch import nn
import torch

from mads_hackathon import datasets, metrics
import mltrainer
import tomllib

class HeartDataset1DBalancer:
    def __init__(
        self,
        path: Path,
        target: str,
        balance: bool = True,
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

        if balance:
            self._balance_classes()

    def _balance_classes(self):
        # Get unique classes and their counts
        unique_classes, class_counts = torch.unique(self.y, return_counts=True)
        max_count = torch.max(class_counts)
        
        # Initialize lists to store balanced data
        balanced_x = []
        balanced_y = []
        
        # Process each class
        for class_idx in unique_classes:
            # Get indices for current class
            class_mask = self.y == class_idx
            class_x = self.x[class_mask]
            class_y = self.y[class_mask]
            
            # Calculate number of repetitions needed
            current_count = class_counts[class_idx]
            multiplier = (max_count // current_count).item()
            remainder = (max_count % current_count).item()
            
            # Add repeated samples
            balanced_x.append(class_x.repeat(multiplier, 1))
            balanced_y.append(class_y.repeat(multiplier))
            
            # Add remainder samples if needed
            if remainder > 0:
                balanced_x.append(class_x[:remainder])
                balanced_y.append(class_y[:remainder])
        
        # Concatenate all balanced data
        self.x = torch.cat(balanced_x, dim=0)
        self.y = torch.cat(balanced_y, dim=0)
        
        # Shuffle the balanced dataset
        indices = torch.randperm(len(self.y))
        self.x = self.x[indices]
        self.y = self.y[indices]


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
    
    



class HeartDataset2DBalancer:
    def __init__(
        self,
        path: Path,
        target: str,
        shape: tuple[int, int] = (16, 12),
        balance: bool = True,
    ) -> None:
        self.df = pd.read_parquet(path)
        self.target = target
        _x = self.df.drop("target", axis=1)
        self.x = torch.tensor(_x.values, dtype=torch.float32)



        y = self.df["target"]
        self.y = torch.tensor(y.values, dtype=torch.int64)


        if balance:
            self._balance_classes()

        # original length is 187, which only allows for 11x17 2D tensors
        # 3*2**6 = 192. This makes it easier to reshape the data
        # it also makes convolutions / maxpooling more predictable
        self.x = torch.nn.functional.pad(self.x, (0, 3 * 2**6 - self.x.size(1))).reshape(
            -1, 1, *shape
        )



    def _balance_classes(self):
        # Get unique classes and their counts
        unique_classes, class_counts = torch.unique(self.y, return_counts=True)
        max_count = torch.max(class_counts)
        
        # Initialize lists to store balanced data
        balanced_x = []
        balanced_y = []
        
        # Process each class
        for class_idx in unique_classes:
            # Get indices for current class
            class_mask = self.y == class_idx
            class_x = self.x[class_mask]
            class_y = self.y[class_mask]
            
            # Calculate number of repetitions needed
            current_count = class_counts[class_idx]
            multiplier = (max_count // current_count).item()
            remainder = (max_count % current_count).item()
            
            # Add repeated samples
            balanced_x.append(class_x.repeat(multiplier, 1))
            balanced_y.append(class_y.repeat(multiplier))
            
            # Add remainder samples if needed
            if remainder > 0:
                balanced_x.append(class_x[:remainder])
                balanced_y.append(class_y[:remainder])
        
        # Concatenate all balanced data
        self.x = torch.cat(balanced_x, dim=0)
        self.y = torch.cat(balanced_y, dim=0)
        
        # Shuffle the balanced dataset
        indices = torch.randperm(len(self.y))
        self.x = self.x[indices]
        self.y = self.y[indices]


    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __repr__(self) -> str:
        return f"Heartdataset2D (#{len(self)})"