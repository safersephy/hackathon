import torch
from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType

flowersfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
streamers = flowersfactory.create_datastreamer(batchsize=32)


from torchvision import transforms

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


import torchvision
from torchvision.models import ResNet18_Weights

resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)


for name, param in resnet.named_parameters():
    param.requires_grad = False


print(type(resnet.fc))
in_features = resnet.fc.in_features
in_features

import torch.nn as nn

resnet.fc = nn.Sequential(
    nn.Linear(in_features, 5)
    # nn.Linear(in_features, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 5)
)


from mltrainer import metrics

accuracy = metrics.Accuracy()

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


from torch import optim

optimizer = optim.SGD
scheduler = optim.lr_scheduler.StepLR

from mltrainer import ReportTypes, TrainerSettings

settings = TrainerSettings(
    epochs=30,
    metrics=[accuracy],
    logdir="modellogs/flowers",
    train_steps=len(train),
    valid_steps=len(valid),
    reporttypes=[ReportTypes.TENSORBOARD],
    optimizer_kwargs={"lr": 0.1, "weight_decay": 1e-05, "momentum": 0.9},
    scheduler_kwargs={"step_size": 10, "gamma": 0.1},
    earlystop_kwargs=None,
)
settings

# note: this will be very slow without acceleration!
# trainer.loop()
