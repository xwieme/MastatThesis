import os
from pathlib import Path
from typing import List

import torch


class EarlyStopping:
    """
    Stop training if the score is worse for a specified amount of steps than
    the current best obtained score.

    :param data_dir: directory where the model is saved
    :param model_name: filename of the saved model, the suffix '_early_stop.pt'
        will be added
    :param patience: number of steps after which the training is stopped if the
        score is worse than the current best score. Resets when a new best
        score is obtained.
    :param mode: specifies if lower is better or higher is better (default is lower)
    """

    def __init__(
        self, data_dir: str, model_name: str, patience: int, mode: str = "lower"
    ):
        # Make sure the save directory exists, create if not
        if not os.path.exists(data_dir):
            Path(data_dir).mkdir(parents=True)

        self.save_path = os.path.join(data_dir, f"{model_name}_early_stop.pt")

        self.patience = patience
        self.counter = 0

        self.best_score = None

        # Specify if a better score is lower or higher
        self._isScoreBetter = self._lower if mode == "lower" else self._higher

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Update best_score or increase counter depending on score
        """

        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), self.save_path)

        elif self._isScoreBetter(score):

            self.best_score = score
            torch.save(model.state_dict(), self.save_path)
            self.counter = 0

        else:
            self.counter += 1

        return self.counter == self.patience

    def _lower(self, score):
        """
        A lower score is better
        """
        return score < self.best_score 

    def _higher(self, score):
        """
        A higher score is better
        """
        return score > self.best_score


def loadModels(
    model: torch.nn.Module, paths: list, device: str = "cpu"
) -> List[torch.nn.Module]:
    """
    Load the specified models

    :param model: model architecture
    :param paths: list of paths where the models are saved
    :param device: load the models on cpu or gpu (default is cpu)
    """

    models = []

    for path in paths:
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        model.eval()
        models.append(model)

    return models
