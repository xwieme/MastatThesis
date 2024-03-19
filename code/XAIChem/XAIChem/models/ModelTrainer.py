import time
from collections import defaultdict
from typing import Callable, Dict

import numpy as np
import pandas as pd
import torch
import wandb
from torch.optim import Optimizer
from torch_geometric.loader.dataloader import DataLoader

from ..handlers import EarlyStopping


class ModelTrainer:
    """
    Class to train a machine learning model and upload training histroy to wandb
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, config: dict):
        self._model = model
        self._device = device
        self._config = config
        self._epoch = 0

        # Initialize empty list to store model predictions and true
        # labels which can be used to compute various evaluation metrics
        self._predictions = []
        self._labels = []

    def reset(self) -> None:
        """
        Reset trainer parameters
        """
        self._epoch = 0

    def trainEpoch(
        self, loader: DataLoader, criterion: Callable, optimizer: Optimizer
    ) -> None:
        """
        Perform one pass of all training data. Parameters are optimized
        after each bach iteration.

        :param loader: batches of training data
        :param criterion: loss function used to optimize the model
            parameters
        :param optimizer: optimization function used to optimize the
            model parameters
        """
        self._model.train()

        # Iterate through the batches
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
        model_type: str = "binary-classification",
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics of the current model using
        the given data set. Returns a dictionairy containing
        performance metric name and its corresponding value.

        :param loader: batches of data
        :param criterion: loss function used to optimize the model
            parameters
        :param metics: dictionairy of functions to compute additional
            performance metrics, e.g. R2, accuracy, F1, ...
            The matrix function should take the true labels as first
            argument and predictions as second argument.
        """

        self._model.eval()

        # Disable gradients to save memory
        with torch.no_grad():
            # Average loss over the batches
            losses = 0

            # Iterate through the batches
            for data in loader:
                data.to(self._device)

                pred = self._model(data)
                losses += criterion(pred, data.y.view(-1, 1)).item()

                if model_type == "prediction":
                    self._predictions.extend(pred.view(1, -1).tolist()[0])

                elif model_type == "binary-classification":
                    self._predictions.extend(torch.round(pred).view(1, -1).tolist()[0])

                self._labels.extend(data.y.tolist())

            evaluation = {"loss": losses / len(loader)}

            if metrics is not None:
                for metric_name, metric in metrics.items():
                    evaluation[metric_name] = metric(self._labels, self._predictions)

        self._predictions = []
        self._labels = []

        return evaluation

    def train(
        self,
        loaders: Dict[str, DataLoader],
        criterion: Callable,
        optimizer: Optimizer,
        epochs: int,
        save_path: str,
        metrics: Dict[str, Callable] | None = None,
        early_stop: EarlyStopping | None = None,
        log: bool = False,
        wandb_project: str | None = None,
        wandb_group: str | None = None,
        wandb_name: str | None = None,
    ) -> None:
        """
        Train the given ML model using the specified criterion and optimizer.

        :param loaders: a dictionairy of DataLoader where the key is the name
        :param criterion: loss function used during training
        :param optimizer: optimizer used to train the ML model
        :param epochs: maximum number of full data iterations allowed
        :param save_path: file location where trained model is saved
        :param metrics: dictionairy of names (strings) and sklearn.metric functions (default is None)
        :param early_stop: Determines when to stop before the number of epochs is reached
            A validation DataLoader must be present if early_stop is used. (default is None)
        :param log: print progress or not (default is False, i.e. no printing)
        :param wandb_project: if specified, upload model progress to wandb with given project name (default is None)
        :param wandb_group: group runs in wandb (default is None)
        :param wandb_name: display name of the run in wandb dashboard (default is None, i.e. a random name will be generated)
        """

        # Store model progress for each dataloader
        evaluations = defaultdict(list)

        # Log training time
        start_time = time.time()

        for epoch in range(epochs):
            self._epoch = epoch

            # Train one epoch using the training data set
            self.trainEpoch(loaders["train"], criterion, optimizer)

            # Log early stop progress
            if early_stop is not None:
                evaluations["early_stop_count"].append(early_stop.counter)
                evaluations["best_score"].append(early_stop.best_score)

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
                    evaluations[f"{loader_name}_{metric}"].append(value)

            # Log passed time and epoch
            evaluations["time"].append(time.time() - start_time)
            evaluations["epoch"].append(epoch)

            if log:
                # Get a list of the metric name together with its
                # latest value
                content = list(
                    zip(
                        evaluations.keys(),
                        np.asarray(list(evaluations.values()))[:, -1],
                    )
                )

                # Print each metric next to each other, add line
                # break for next log statement
                for metric in content:
                    if metric[1] is not None:
                        print(f"{metric[0]}: {metric[1]:<8.4f}", end="\t")
                print("\n", end="")

            # If early stop is reach, stop training
            if early_stop is not None and early_stop(
                evaluations[f"validation_{self._config['early_stop']['metric']}"][
                    self._epoch
                ],
                self._model,
            ):
                break

        # Safe latest model
        torch.save(self._model.state_dict(), save_path)

        # if a wandb project name is given, upload model progress to wandb
        if wandb_project is not None:
            wandb.init(
                project=wandb_project,
                group=wandb_group,
                name=wandb_name,
                config=self._config,
            )

            # Every row of the pandas dataframe represents one training step
            df = pd.DataFrame.from_dict(evaluations)
            df.apply(lambda row: wandb.log(row.to_dict()), axis=1)

            wandb.finish()
