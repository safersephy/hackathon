
import torch
import torch.nn as nn
import torchvision
import mlflow
import mlflow.pytorch

from pathlib import Path
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune import TuneConfig
from ray.tune.search.hyperopt import HyperOptSearch

from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType
from torch import optim
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from torch.nn import CrossEntropyLoss
from torcheval.metrics import MulticlassAccuracy
from tytorch.trainer import EarlyStopping, Trainer
from tytorch.utils.mlflow import set_best_run_tag_and_log_model, set_mlflow_experiment

tuningmetric = "valid_loss"
tuninggoal = "min"
n_trials = 60
n_epochs = 30

flowersfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)


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

experiment_name = set_mlflow_experiment("tune")
params = {
    "lr": tune.loguniform(1e-4,1e-2),
    "n_epochs": n_epochs
}



def tune_func(config: dict) -> None:


    streamers = flowersfactory.create_datastreamer(batchsize=32)
    trainprocessor = AugmentPreprocessor(data_transforms["train"])
    validprocessor = AugmentPreprocessor(data_transforms["val"])

    traindataset = streamers["train"]
    validdataset = streamers["valid"]
    traindataset.preprocessor = trainprocessor
    validdataset.preprocessor = validprocessor

    trainstreamer = traindataset.stream()
    validstreamer = validdataset.stream()


    resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for name, param in resnet.named_parameters():
        param.requires_grad = False

    in_features = resnet.fc.in_features

    resnet.fc = nn.Sequential(
        nn.Linear(in_features, 5)
    )

    optimizer = optim.SGD(
        resnet.parameters(), 
        lr= config["lr"],
        weight_decay=1e-05,
        momentum=0.9
        )
    trainer = Trainer(
        model=resnet,
        loss_fn=CrossEntropyLoss(),
        metrics=[MulticlassAccuracy()],
        optimizer=optimizer,
        device=device,
        quiet=True,
        train_steps=len(traindataset),
        valid_steps=len(validdataset),    
        lrscheduler=optim.lr_scheduler.StepLR(
            optimizer=optimizer, 
            step_size=10, 
            gamma=0.1),
    )
    
    mlflow.log_params(config)

    n_epochs = int(params.get("n_epochs", 10))  # type: ignore

    trainer.fit(n_epochs, trainstreamer, validstreamer)


tuner = tune.Tuner(
    tune.with_resources(tune_func, {"cpu": 10}),
    param_space=params,
    tune_config=TuneConfig(
        mode=tuninggoal,
        search_alg=HyperOptSearch(),
        metric=tuningmetric,
        num_samples=n_trials,
        max_concurrent_trials=1,
    ),
    run_config=train.RunConfig(
        storage_path=Path("./ray_tuning_results").resolve(),
        name=experiment_name,
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri=mlflow.get_tracking_uri(),
                experiment_name=experiment_name,
                save_artifact=True,
            )
        ],
    ),
)
results = tuner.fit()

best_result = results.get_best_result(tuningmetric, tuninggoal)
model = params["model_class"](best_result.config)  # type: ignore
model.load_state_dict(torch.load(Path(best_result.checkpoint.path) / "model.pth"))
set_best_run_tag_and_log_model(experiment_name, model, tuningmetric, tuninggoal)

