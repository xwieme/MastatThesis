import re
from collections import defaultdict
from io import BytesIO
from typing import List, Set, Tuple

from PIL import Image, PngImagePlugin
from plotly.express.colors import sample_colorscale
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

from .structures import getSubstructureBondIds


def getRGBA(
    values: List[float], colorscale: str = "rdbu", alpha: float = 1
) -> List[Tuple[int]]:
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
) -> PngImagePlugin.PngImageFile:
    drawer = Draw.MolDraw2DCairo(350, 300)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.annotationFontScale = 0.8
    options.dummiesAreAttachments = True
    options.addAtomIndices = show_atom_indices
    options.addBondIndices = show_bond_indices

    rdDepictor.Compute2DCoords(molecule)
    rdDepictor.StraightenDepiction(molecule)

    # Convert given values to a rgba color scale using plotly
    colors = getRGBA(atoms_highlight_values.values(), colorscale)

    highlight_atoms = defaultdict(list)
    highlight_bonds = defaultdict(list)

    for i, item in enumerate(atoms_highlight_values.items()):
        substructure, value = item

        molecule.GetAtomWithIdx(substructure[0]).SetProp("atomNote", str(value))

        for atom_id in substructure:
            highlight_atoms[atom_id].append(colors[i])

        if len(substructure) > 1:
            bond_ids = getSubstructureBondIds(molecule, substructure)
            for bond_id in bond_ids:
                highlight_bonds[bond_id].append(colors[i])

    drawer.DrawMoleculeWithHighlights(
        molecule, legend, dict(highlight_atoms), dict(highlight_bonds), {}, {}
    )
    drawer.FinishDrawing()

    return Image.open(BytesIO(drawer.GetDrawingText()))
