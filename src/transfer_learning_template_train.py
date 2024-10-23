
import torch
import torch.nn as nn
import torchvision
import mlflow
import mlflow.pytorch
from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType
from torch import optim
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from torch.nn import CrossEntropyLoss
from torcheval.metrics import MulticlassAccuracy
from tytorch.trainer import EarlyStopping, Trainer
from tytorch.utils.mlflow import set_mlflow_experiment

flowersfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
streamers = flowersfactory.create_datastreamer(batchsize=32)
n_epochs = 30

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


resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

for name, param in resnet.named_parameters():
    param.requires_grad = False

in_features = resnet.fc.in_features

resnet.fc = nn.Sequential(
    nn.Linear(in_features, 5)
)


if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"
    logger.warning(
        "This model will take 15-20 minutes on CPU. Consider using accelaration, eg with google colab (see button on top of the page)"
    )
logger.info(f"Using {device}")



optimizer = optim.SGD(
    resnet.parameters(), 
    lr= 0.1,
    weight_decay=1e-05,
    momentum=0.9
    )

trainer = Trainer(
    model=resnet,
    loss_fn=CrossEntropyLoss(),
    metrics=[MulticlassAccuracy()],
    optimizer=optimizer,
    device=device,
    train_steps=len(train),
    valid_steps=len(valid),    
    lrscheduler=optim.lr_scheduler.StepLR(
        optimizer=optimizer, 
        step_size=10, 
        gamma=0.1),
)

set_mlflow_experiment("train")
with mlflow.start_run():
    #mlflow.log_params(params)
    trainer.fit(n_epochs, trainstreamer, validstreamer)

    mlflow.pytorch.log_model(resnet, artifact_path="logged_models/model")
mlflow.end_run()
