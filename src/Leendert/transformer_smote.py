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
import numpy as np
import torch.optim as optim
uri, dev = config()

datadir = Path('src\Leendert\data\hackathon-data')
trainfile = (datadir / "heart_big_train_bal.parq").resolve()
validfile = (datadir / "heart_big_valid_bal.parq").resolve()
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

traindataset = datasets.HeartDataset1DBalancer(trainfile, target="target", balance=False)
validdataset = datasets.HeartDataset1DBalancer(validfile, target="target", balance=False)

config = Config(
    batchsize=128,
    dropout=0.1,
    input_channels=1,
    hidden=64,
    kernel_size=3,
    stride=1,
    num_heads=4,
    num_blocks=4,
    num_classes=5,
)

trainstreamer = BaseDatastreamer(traindataset, preprocessor = BasePreprocessor(), batchsize=config.batchsize)
validstreamer = BaseDatastreamer(validdataset, preprocessor = BasePreprocessor(), batchsize=config.batchsize)
from torch.optim.lr_scheduler import ReduceLROnPlateau


model = Transformer(config)
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
mlflow.set_experiment("Leendert")

loss_fn = torch.nn.CrossEntropyLoss()

def calculate_cfm(model, teststreamer, device):
    y_true = []
    y_pred = []

    testdata = teststreamer.stream()
    for _ in range(len(teststreamer)):
        X, y = next(testdata)
        X, y = X.to(device), y.to(device)
        yhat = model(X)
        yhat = yhat.argmax(dim=1)  # we get the one with the highest probability
        y_pred.append(yhat.cpu().tolist())
        y_true.append(y.cpu().tolist())

    yhat = [x for y in y_pred for x in y]
    y = [x for y in y_true for x in y]

    cfm = confusion_matrix(y, yhat)
    cfm = cfm / np.sum(cfm, axis=1, keepdims=True)
    return cfm

with mlflow.start_run():
    optimizer = optim.AdamW
    scheduler = ReduceLROnPlateau
    settings = TrainerSettings(
        epochs=15,
        metrics=[accuracy, f1micro, f1macro, precision, recall],
        logdir="logs/heart1D",
        train_steps=len(trainstreamer),
        valid_steps=len(validstreamer),
        reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
        scheduler_kwargs={"factor": 0.1, "patience": 10},
        earlystop_kwargs={'save': True},
        optimizer_kwargs={"lr": 5e-4, "weight_decay": 1e-5},
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
        scheduler=scheduler,
        device=device,
        )
    trainer.loop()
    cfm = calculate_cfm(model, validstreamer, device)
    for i, tp in enumerate(np.diag(cfm)):
        mlflow.log_metric(f"TP_{i}", tp)