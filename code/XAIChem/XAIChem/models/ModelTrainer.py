import time
from collections import defaultdict
from typing import Callable, Dict

import pandas as pd
import torch
import wandb
from torch.optim import Optimizer
from torch_geometric.loader.dataloader import DataLoader
from XAIChem.handlers import EarlyStopping


class ModelTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ):
        self._model = model
        self._epoch = 0
        self._device = device

        # Initialize empty list to store model predictions and true
        # labels which can be used to compute various evaluation metrics
        self._predictions = list()
        self._labels = list()

    def reset(self) -> None:
        """
        Reset trainer parameters
        """
        self._epoch = 0

    def trainEpoch(
        self, loader: DataLoader, criterion: Callable, optimizer: Optimizer
    ) -> None:
        self._model.train()

        for data in loader:
            data.to(self._device)

            out = self._model(data)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def evaluate(
        self,
        loader: DataLoader,
        criterion: Callable,
        metrics: Dict[str, Callable] | None = None,
    ) -> Dict[str, float]:
        self._model.eval()

        # Disable gradients to save memory
        with torch.no_grad():
            # Compute average batch loss
            losses = 0

            for data in loader:
                data.to(self._device)

                pred = self._model(data)
                losses += criterion(pred, data.y.view(-1, 1)).item()
                self._predictions.extend(pred.view(1, -1).tolist()[0])
                self._labels.extend(data.y.tolist())

            evaluation = {"loss": losses / len(loader)}

            if metrics is not None:
                for metric_name, metric in metrics.items():
                    evaluation[metric_name] = metric(self._labels, self._predictions)

        self._predictions = list()
        self._labels = list()

        return evaluation

    def train(
        self,
        loaders: Dict[str, DataLoader],
        criterion: Callable,
        optimizer: Optimizer,
        epochs: int,
        metrics: Dict[str, Callable] | None = None,
        early_stop: EarlyStopping | None = None,
        log: bool = False,
        wandb_project: str | None = None,
        wandb_group: str | None = None,
        log_filename: str | None = None,
    ) -> None:
        """
        Train the given ML model using the specified criterion and optimizer.

        :param loaders: a dictionairy of DataLoader where the key is the name
        :param criterion: loss function used during training
        :param optimizer: optimizer used to train the ML model
        :param epochs: maximum number of full data iterations allowed
        :param metrics: dictionairy of names (strings) and sklearn.metric functions (default is None)
        :param early_stop: Determines when to stop before the number of epochs is reached
            A validation DataLoader must be present if early_stop is used. (default is None)
        :param log: print progress or not (default is False, i.e. no printing)
        :param wandb_project: if specified, upload model progress to wandb with given project name (default is None)
        :param wandb_group: group runs in wandb (default is None)
        :param log_filename: if specified, write model progess to disk with given path
        """

        # Store model progress for each dataloader
        evaluations = {
            dataset_name: defaultdict(list) for dataset_name in loaders.keys()
        }

        # Log training time
        start_time = time.time()

        for epoch in range(epochs):
            self._epoch = epoch

            # Train one epoch using the training data set
            self.trainEpoch(loaders["train"], criterion, optimizer)

            # Evaluate current model using the provided data sets
            for loader_name, loader in loaders.items():
                # Compute loss and requested metrics
                evaluation = self.evaluate(
                    loader,
                    criterion,
                    metrics,
                )

                # Add loss and requested metrics to model progress dict
                for metric, value in evaluation.items():
                    evaluations[loader_name][metric].append(value)

                evaluations[loader_name]["time"].append(time.time() - start_time)

            if log:
                print(
                    {
                        loader_name: {
                            metric: values[self._epoch]
                            for metric, values in evaluation.items()
                        }
                        for loader_name, evaluation in evaluations.items()
                    }
                )

            if early_stop is not None and early_stop(
                evaluations["validation"]["loss"][self._epoch], self._model
            ):
                break

        # if a wandb project name is given, upload model progress to wandb
        if wandb_project is not None:
            wandb.init(project=wandb_project, group=wandb_group)
            wandb.log(evaluations.values())
            wandb.finish()

        # If a log_filename is given, write model progress to disk
        if log_filename is not None:
            for dataset_name, model_progress in evaluations.items():
                df = pd.DataFrame(model_progress)
                df.to_json(f"{log_filename}_{dataset_name}.json")
