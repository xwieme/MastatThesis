from collections.abc import Callable
from typing import Dict

import numpy as np
import torch
from torch_geometric.loader.dataloader import DataLoader

class ModelTrainer():

    def __init__(self, model: torch.nn.Module, device: torch.device , init_val_loss: float = np.inf):
        """
        :param init_val_loss: initial value of validation loss
            only used for early stopping
        """
        self.model = model
        self.epoch = 0
        self.val_loss = init_val_loss
        self.device = device

        # Initialize empty list to store model predictions and true 
        # labels which can be used to compute various evaluation metrics
        self.predictions = list()
        self.labels = list()

    def reset(self, init_val_loss: float = np.inf) -> None:
        """
        Reset trainer parameters

        :param init_val_loss: initial value of validation loss
            only used for early stopping
        """
        self.epoch = 0
        self.val_loss = init_val_loss

    def trainEpoch(self, loader: DataLoader, criterion, optimizer):
        self.model.train()

        for data in loader:
            data.to(self.device)

            out = self.model(data)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    def evaluateEpoch(self, loader, criterion, metrics: Dict[str, Callable] | None = None) -> Dict[str, float]:
        self.model.eval()

        # Disable gradients to save memory
        with torch.no_grad():
       
            # Compute average batch loss 
            losses = 0

            for data in loader:
                data.to(self.device)
                
                pred = self.model(data)
                losses += criterion(pred, data.y.view(-1, 1))
        
                self.predictions.extend(pred.view(1, -1))
                self.labels.extend(data.y)

            evaluation = { "loss": losses / len(loader) }

            if metrics is not None:
                for metric_name, metric in metrics.items():
                    evaluation[metric_name] = metric(self.labels, self.predictions)
        
        return evaluation 

    def _trainStep(self, loaders, criterion, optimizer, metrics, evaluations, log):

        # Train one epoch using the training data set
        self.trainEpoch(loaders["train"], criterion, optimizer)

        # Evaluate current model using the provided data sets 
        evaluations[self.epoch] = dict()
        for loader_name, loader in loaders.items():
            evaluations[self.epoch][loader_name] = self.evaluateEpoch(loader, criterion, metrics)

        if log:
            print(evaluations[self.epoch])

    def train(self, loaders: Dict[str, DataLoader], criterion, optimizer, epochs: int, metrics: Dict[str, Callable] | None = None, early_stop: None = None, log: bool = False) -> None:
      
        evaluations = dict()

        if early_stop is None:

            for epoch in range(epochs):
                
                self.epoch = epoch 
                self._trainStep(loaders, criterion, optimizer, metrics, evaluations, log)

        else:

            while self.epoch < epochs and not early_stop(self.val_loss, self.model):
                
                self._trainStep(loaders, criterion, optimizer, metrics, evaluations, log)
                self.val_loss = evaluations[self.epoch]["validation"]["loss"]
                self.epoch += 1











                
