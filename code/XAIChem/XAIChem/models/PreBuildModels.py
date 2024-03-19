import torch.nn.functional as F
import yaml
from torch_geometric.nn import FastRGCNConv, RGCNConv

from ..features import getNumAtomFeatures
from ..utils import set_seed
from . import MLP, RGCN, MolecularPropertyPredictor, WeightedSum


def rgcnWuEtAll(config_file: str, args: list, **kwargs):
    """
    Build a machine learning model to predict
    molecular property using the architecture
    proposed by Wu et. all. (:TODO: doi).

    :param config_file: path to yaml file containing
        the model specifications
    :param args: keys in config_file that contain
        variables
    :param kwargs: name and value of variables used
        in config_file

    """

    with open(config_file, "r") as f:
        config = yaml.safe_load(f.read())

    # Evaluate all variables
    for key in args:
        config[key] = eval(config[key])

    set_seed(config["seed"])

    # Construct a MolecularPropertiePredicor
    gnn = RGCN(
        getNumAtomFeatures(),
        config["RGCN"]["num_units"],
        rgcn_conv=RGCNConv,
        dropout_rate=config["RGCN"]["dropout_rate"],
        use_batch_norm=config["RGCN"]["use_batch_norm"],
        use_residual=config["RGCN"]["use_residual"],
        num_bases=config["RGCN"]["num_bases"],
    )

    molecular_embedder = WeightedSum(config["RGCN"]["num_units"][-1])

    mlp = MLP(
        config["MLP"]["num_layers"],
        config["MLP"]["dropout_rate"],
        config["RGCN"]["num_units"][-1],
        config["MLP"]["num_units"],
    )

    # Combine all parts, apply sigmoid function to model output to obtain
    # probabilities if the model is a classification.
    if config["is_classification"]:
        model = MolecularPropertyPredictor(gnn, molecular_embedder, mlp, F.sigmoid)

    else:
        model = MolecularPropertyPredictor(gnn, molecular_embedder, mlp)

    return model, config
