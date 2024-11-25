from mads_datasets.base import BaseDatastreamer
from mltrainer.preprocessors import BasePreprocessor
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch import nn
import torch
from mads_hackathon.models import TransformerConfig as Config
from torchinfo import summary
from mads_hackathon import datasets, metrics
import mltrainer
import tomllib
from mads_hackathon.metrics import caluclate_cfm
from config import config
from mltrainer import Trainer, TrainerSettings, ReportTypes
from dataclasses import asdict
from mads_hackathon.models import Transformer
uri, dev = config()

datadir = Path('src\Leendert\data\hackathon-data')
trainfile = (datadir / "heart_big_train.parq").resolve()
validfile = (datadir / "heart_big_valid.parq").resolve()
trainfile.exists(), validfile.exists()

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device = "cuda:0"
    print("using cuda")
else:
    device = "cpu"
    print("using cpu")

traindataset = datasets.HeartDataset1D(trainfile, target="target")
validdataset = datasets.HeartDataset1D(validfile, target="target")

config = Config(
    batchsize=128,
    dropout=0.1,
    input_channels=1,
    hidden=64,
    kernel_size=3,
    stride=1,
    num_heads=2,
    num_blocks=2,
    num_classes=5,
)

trainstreamer = BaseDatastreamer(traindataset, preprocessor = BasePreprocessor(), batchsize=config.batchsize)
validstreamer = BaseDatastreamer(validdataset, preprocessor = BasePreprocessor(), batchsize=config.batchsize)

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[64, 128], dropout=0.2):
        super(RNNModel, self).__init__()
        
        self.masking = None  # PyTorch handles masking differently
        self.hidden_sizes = hidden_sizes
        
        # Create Bidirectional GRU layers
        self.gru_layers = nn.ModuleList()
        input_size_gru = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            gru_layer = nn.GRU(
                input_size=input_size_gru,
                hidden_size=hidden_size,
                batch_first=True,
                dropout=dropout if i < len(hidden_sizes) - 1 else 0,
                bidirectional=True
            )
            self.gru_layers.append(gru_layer)
            input_size_gru = hidden_size * 2  # * 2 because of bidirectional
        
        # Dense layers
        self.dense1 = nn.Linear(hidden_sizes[-1] * 2, 64)  # * 2 because of bidirectional
        self.dense2 = nn.Linear(64, 16)
        self.output = nn.Linear(16, 5)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Process through GRU layers
        current_layer = x
        for i, gru_layer in enumerate(self.gru_layers):
            current_layer, _ = gru_layer(current_layer)
            if i < len(self.gru_layers) - 1:
                # Keep all sequences for non-final layers
                pass
            else:
                # For the last GRU layer, only keep the last output
                current_layer = current_layer[:, -1, :]
        
        # Dense layers
        current_layer = self.relu(self.dense1(current_layer))
        current_layer = self.relu(self.dense2(current_layer))
        logits = self.relu(self.output(current_layer))
        
        return logits
    
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Example usage:
def create_model(sequence_length=187):
    model = RNNModel(
        input_size=1,
        hidden_sizes=[64, 128],
        dropout=0.2
    )
    
    print('Parameter count:', model.parameter_count())
    
    # If you want to see the model structure:
    print("\nModel structure:")
    print(model)
    
    # Test the model with dummy data
    batch_size = 32
    x = torch.randn(batch_size, sequence_length, 1)
    output = model(x)
    print("\nOutput shape:", output.shape)  # Should be (batch_size, 5)
    
    return model


model = create_model()
model.to(device)
x, y = next(trainstreamer.stream())
x.shape, y.shape
summary(model, input_size=x.shape)

f1micro = metrics.F1Score(average='micro')
f1macro = metrics.F1Score(average='macro')
precision = metrics.Precision('micro')
recall = metrics.Recall('macro')
accuracy = metrics.Accuracy()

import mlflow
mlflow.set_tracking_uri(uri)
mlflow.set_experiment("Test_script")

loss_fn = torch.nn.CrossEntropyLoss()

with mlflow.start_run():
    optimizer = torch.optim.Adam

    settings = TrainerSettings(
        epochs=15,
        metrics=[accuracy, f1micro, f1macro, precision, recall],
        logdir="logs/heart1D",
        train_steps=len(trainstreamer),
        valid_steps=len(validstreamer),
        reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
        scheduler_kwargs=None,
        earlystop_kwargs=None
    )

    mlflow.set_tag("model", "Transfomer")
    mlflow.set_tag("dataset", "heart1D")
    mlflow.set_tag("dev", dev)
    mlflow.log_params(asdict(config))
    mlflow.log_param("epochs", settings.epochs)
    mlflow.log_param("optimizer", str(optimizer))
    mlflow.log_param("scheduler", "None")
    mlflow.log_param("earlystop", "None")
    mlflow.log_params(settings.optimizer_kwargs)

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optimizer,
        traindataloader=trainstreamer.stream(),
        validdataloader=validstreamer.stream(),
        scheduler=None,
        device=device,
        )
    trainer.loop()