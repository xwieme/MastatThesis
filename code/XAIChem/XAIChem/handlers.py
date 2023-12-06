import os
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
        self.path = os.path.join(data_dir, f"{model_name}_early_stop.pt")

        self.patience = patience
        self.counter = 0

        self.best_score = None
        self._isScoreBetter = (
            lambda score: score < self.best_score
            if mode.lower() == "lower"
            else lambda score: score > self.best_score
        )

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), self.path)

        elif self._isScoreBetter(score):
            self.best_score = score
            torch.save(model.state_dict(), self.path)
            self.counter = 0

        else:
            self.counter += 1

        return self.counter == self.patience


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
