import torch


def getEdgeType(edge_attr: torch.tensor):
    """
    Convert edge feature vector to an integer representing the edge type.
    """

    return (
        torch.where(edge_attr[:4] == 1)[0] +\
        edge_attr[4:5] * 4 +\
        edge_attr[5:6] * 8 +\
        torch.where(edge_attr[6:] == 1)[0]
    )


def getEdgeTypes(edge_attr: torch.tensor):
    """
    Convert the edge feature matrix to one dimensional 
    tensor of type long representing the type of the edge.
    """

    return (
        torch.where(edge_attr[:, :4] == 1)[1] +\
        edge_attr[:, 4:5].view(1, -1) * 4 +\
        edge_attr[:, 5:6].view(1, -1) * 8 +\
        torch.where(edge_attr[:, 6:] == 1)[1]
    )
