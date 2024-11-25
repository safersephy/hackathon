import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class CNNConfig:
    matrixshape: tuple
    batchsize: int
    input_channels: int
    hidden: int
    kernel_size: int
    maxpool: int
    num_layers: int
    num_classes: int


@dataclass
class TransformerConfig:
    batchsize: int
    dropout: float
    input_channels: int
    hidden: int
    kernel_size: int
    stride: int
    num_heads: int
    num_blocks: int
    num_classes: int


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class CNN(nn.Module):
    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        input_channels = config.input_channels
        kernel_size = config.kernel_size
        hidden = config.hidden
        # first convolution
        self.convolutions = nn.ModuleList(
            [
                ConvBlock(input_channels, hidden, kernel_size),
            ]
        )

        # additional convolutions
        pool = config.maxpool
        num_maxpools = 0
        for i in range(config.num_layers):
            self.convolutions.extend(
                [ConvBlock(hidden, hidden, kernel_size), nn.ReLU()]
            )
            # every two layers, add a maxpool
            if i % 2 == 0:
                num_maxpools += 1
                self.convolutions.append(nn.MaxPool2d(pool, pool))

        # let's try to calculate the size of the linear layer
        # please note that changing stride/padding will change the logic
        matrix_size = (config.matrixshape[0] // (pool**num_maxpools)) * (
            config.matrixshape[1] // (pool**num_maxpools)
        )
        print(f"Calculated matrix size: {matrix_size}")
        print(f"Caluclated flatten size: {matrix_size * hidden}")

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(matrix_size * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convolutions:
            x = conv(x)
        x = self.dense(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_seq_len, d_model)
        # batch, seq_len, d_model
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        # feel free to change the input parameters of the constructor
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.hidden),
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden)
        self.layer_norm2 = nn.LayerNorm(config.hidden)

    def forward(self, x):
        identity = x.clone()  # skip connection
        x, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + identity)  # Add & Norm skip
        identity = x.clone()  # second skip connection
        x = self.ff(x)
        x = self.layer_norm2(x + identity)  # Add & Norm skip
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=config.input_channels,
            out_channels=config.hidden,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=1,
        )
        self.pos_encoder = PositionalEncoding(config.hidden, config.dropout)

        # Create multiple transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_blocks)]
        )

        self.out = nn.Linear(config.hidden, config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # streamer:         (batch, seq_len, channels)
        # conv1d:           (batch, channels, seq_len)
        # pos_encoding:     (batch, seq_len, channels)
        # attention:        (batch, seq_len, channels)
        x = x.transpose(1, 2)  # flip channels and seq_len for conv1d
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # flip back to seq_len and channels
        x = self.pos_encoder(x)

        # Apply multiple transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = x.mean(dim=1)  # Global Average Pooling
        x = self.out(x)
        return x
