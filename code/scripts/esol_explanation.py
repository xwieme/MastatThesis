from itertools import chain, combinations
from typing import Iterable
from collections import deque

import pandas as pd
import numpy as np

import XAIChem 


def powerset(A: Iterable):
    """
    Generate the powerset of A.

    Example: powerset([1, 2, 3]) results in 
    [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """

    A = list(A)
    return chain.from_iterable(combinations(A, r) for r in range(len(A)+1))


def isConnected(i, j, g):
    return (i, j) in g


def adjacencyMatrix(N, g):

    A = np.zeros((max(N), max(N)))

    # Iterate throug every edge, set an 
    # element to one if an edge exists between 
    # its corresponding indices
    for i, j in g:
        A[i-1, j-1] = 1
        A[j-1, i-1] = 1

    return A


def path(N, g, i, j):
    """
    Check if there exists a path for i to j in the graph 
    defined by the edges set g.
    """

    assert i != j, "Self loops are not defined"

    A = adjacencyMatrix(N, g)

    to_visit = deque(*np.where(A[i-1, :] == 1))
    visited = {i}

    current_index = to_visit.pop() + 1

    while current_index != j or visited == N:
        # Mask current index as visited
        visited.add(current_index)

        # Get all connected vertices to the current vertex
        # If they are not visited yet, assign to to_visit
        for neighbor in np.where(A[current_index-1, :] == 1)[0]:
            if neighbor + 1 not in visited:
                to_visit.append(neighbor)

        # Go to the next vertex
        current_index = to_visit.pop() + 1

    return current_index == j


def partition(S, g):
    """
    Create a set of connected components of S on the graph defined 
    by the edge set g.
    """

    out = [] 

    for j in S:
        component = [j]
        for i in S:
            if i != j and path(N, g, i, j):
                component.append(i)

        out.append(tuple(component))
    
    return set(out)


def createMc(N, t):

    M = np.zeros(shape=(2**len(N) - 1, 2**len(N) - 1))
    all_coalitions = np.asarray(list(powerset(N)), dtype=object)[1:]

    # Diagonal elements
    for i, S in enumerate(all_coalitions):
        M[i, i] = 1 - (len(N) - len(S)) * t

        for j, T in enumerate(all_coalitions):

            if len(T) == 1 and not set(T).issubset(set(S)):
                M[i, j] = -t
            
            elif len(S) + 1 == len(T) and set(S).issubset(set(T)):
                M[i, j] = t

    return M


def createPg():
    pass


if __name__ == "__main__":

    import doctest 
    doctest.testmod()

    # Get model architecture and configuration
#    model, config = XAIChem.models.PreBuildModels.rgcnWuEtAll(
#        "esol_reproduction.yaml",
#        ["seed"],
#        model_id=0
#    )
#
#    # Load trained models 
#    paths = [
#        f"../../data/ESOL/trained_models/ESOL_rgcn_model_{i}_early_stop.pt"
#        for i in range(10)
#    ]
#    models = XAIChem.loadModels(model, paths)
#
#    # Load data for explanation
    molecules = pd.read_csv("../../data/ESOL/ESOL.csv")
#
#    masks = XAIChem.substructures.functionalGroupMasks(molecules.smiles)
#    attributions = XAIChem.attribution.difference(models, masks)
#
#    print(attributions.head())

    # molecule = XAIChem.createDataObjectFromSmiles(molecules["smiles"][0], np.inf)

    N = {1, 2, 3}
    g = {(1, 2), (2, 3)}

    M = createMc(N, 1)
    print(M)

    print(partition(N, g))
    
    
