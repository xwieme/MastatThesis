import random
import numpy as np
import torch


def getEdgeType(edge_attr: torch.tensor):
    """
    Convert edge feature vector to an integer representing the edge type.
    """

    return (
        torch.where(edge_attr[:4] == 1)[0]
        + edge_attr[4:5] * 4
        + edge_attr[5:6] * 8
        + torch.where(edge_attr[6:] == 1)[0]
    )


def getEdgeTypes(edge_attr: torch.tensor):
    """
    Convert the edge feature matrix to one dimensional
    tensor of type long representing the type of the edge.
    """

    return (
        torch.where(edge_attr[:, :4] == 1)[1]
        + edge_attr[:, 4:5].view(1, -1) * 4
        + edge_attr[:, 5:6].view(1, -1) * 8
        + torch.where(edge_attr[:, 6:] == 1)[1]
    )


def set_seed(seed):
    """
    Sets the seed of python, numpy and pytorch
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
