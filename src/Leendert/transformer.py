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
mlflow.set_experiment("Test_script")


loss_fn = torch.nn.CrossEntropyLoss()

with mlflow.start_run():
    optimizer = torch.optim.Adam

    settings = TrainerSettings(
        epochs=5,
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
        )
    trainer.loop()