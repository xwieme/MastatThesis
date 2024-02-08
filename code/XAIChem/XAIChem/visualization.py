import base64
import re
from collections import defaultdict
from io import BytesIO
from typing import List, Set, Tuple

import numpy as np
from PIL import Image, PngImagePlugin
from plotly.express.colors import sample_colorscale
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

from .structures import getSubstructureBondIds


def getRGBA(
    values: List[float] | np.ndarray, colorscale: str = "rdbu", alpha: float = 1
) -> List[Tuple[float, ...]]:
    """
    Converts a list of floats to a rgb color using plotly color scales.

    :param value: float value to convert to an rgb color
    :param colorscale: name of a plotly color scale
    :param alpha: transparancy degree, zero is completely transparant, one
        means not transparancy
    """
    if not isinstance(values, list):
        values = list(values)

    # The plotly color scales are defined between 0 and 1, but the values range
    # from -1 to 1.
    values = [value / 2 + 0.5 for value in values]

    return [
        tuple([int(value) / 255 for value in re.findall("\d+", color)] + [alpha])
        for color in sample_colorscale(colorscale, values)
    ]


def showMolecule(
    molecule: Chem.rdchem.Mol,
    atoms_highlight_values: dict = {},
    legend: str = "",
    show_atom_indices: bool = False,
    show_bond_indices: bool = False,
    colorscale: str = "rdbu",
    width: int = 350,
    height: int = 300,
) -> PngImagePlugin.PngImageFile:
    drawer = Draw.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.annotationFontScale = 0.8
    options.dummiesAreAttachments = True
    options.addAtomIndices = show_atom_indices
    options.addBondIndices = show_bond_indices
    # options.highlightBondWidthMultiplier = 100

    rdDepictor.Compute2DCoords(molecule)
    rdDepictor.StraightenDepiction(molecule)

    # Convert given values to a rgba color scale using plotly
    colors = getRGBA(np.tanh(list(atoms_highlight_values.values())), colorscale)

    highlight_atoms = defaultdict(list)
    highlight_bonds = defaultdict(list)
    atom_radia = {}
    bond_radia = {}

    for i, item in enumerate(atoms_highlight_values.items()):
        substructure, value = item

        molecule.GetAtomWithIdx(substructure[0]).SetProp("atomNote", str(value))

        for atom_id in substructure:
            highlight_atoms[atom_id].append(colors[i])
            atom_radia[atom_id] = 1

        if len(substructure) > 1:
            bond_ids = getSubstructureBondIds(molecule, substructure)
            for bond_id in bond_ids:
                highlight_bonds[bond_id].append(colors[i])
                bond_radia[bond_id] = 500000000

    drawer.DrawMoleculeWithHighlights(
        molecule,
        legend,
        dict(highlight_atoms),
        dict(highlight_bonds),
        atom_radia,
        bond_radia,
    )
    drawer.FinishDrawing()

    values = np.zeros(len(molecule.GetAtoms()))
    for substructure, value in atoms_highlight_values.items():
        for atom_id in substructure:
            values[atom_id] = value

    return Image.open(BytesIO(drawer.GetDrawingText()))
