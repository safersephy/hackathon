import tempfile

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torchvision
from dotenv import load_dotenv
from mads_datasets import DatasetFactoryProvider, DatasetType
from torch import optim
from torch.nn import CrossEntropyLoss
from torcheval.metrics import MulticlassAccuracy
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from tytorch.trainer import Trainer
from tytorch.utils.mlflow import set_mlflow_experiment
from tytorch.utils.trainer_utils import get_device

# get environment variables to upload artifacts to central mlflow
load_dotenv()

params = {"n_epochs": 2, "lr": 0.1}

# data prep
flowersfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
streamers = flowersfactory.create_datastreamer(batchsize=32)


data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

flowersfactory.settings.img_size = (500, 500)


class AugmentPreprocessor:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        X = [self.transform(x) for x in X]
        return torch.stack(X), torch.stack(y)


trainprocessor = AugmentPreprocessor(data_transforms["train"])
validprocessor = AugmentPreprocessor(data_transforms["val"])

train = streamers["train"]
valid = streamers["valid"]
train.preprocessor = trainprocessor
valid.preprocessor = validprocessor
trainstreamer = train.stream()
validstreamer = valid.stream()


# load model, freeze the params, and create new tail
resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

for name, param in resnet.named_parameters():
    param.requires_grad = False

in_features = resnet.fc.in_features

resnet.fc = nn.Sequential(nn.Linear(in_features, 5))

# let's train!
device = get_device()

optimizer = optim.SGD(
    resnet.parameters(), lr=params["lr"], weight_decay=1e-05, momentum=0.9
)

trainer = Trainer(
    model=resnet,
    loss_fn=CrossEntropyLoss(),
    metrics=[MulticlassAccuracy()],
    optimizer=optimizer,
    device=device,
    train_steps=len(train),
    valid_steps=len(valid),
    lrscheduler=optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1),
)


set_mlflow_experiment(
    "train", True, tracking_uri="http://madsmlflowwa.azurewebsites.net"
)
with mlflow.start_run():
    mlflow.log_params(params)
    trainer.fit(params["n_epochs"], trainstreamer, validstreamer)

    with tempfile.TemporaryDirectory() as path:
        model_path = f"{path}/resnet18.pth"
        torch.save(resnet.state_dict(), model_path)
        mlflow.log_artifact(path)

mlflow.end_run()
