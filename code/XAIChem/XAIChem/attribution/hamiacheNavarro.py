from functools import lru_cache
from typing import Callable, Iterable

import numpy as np

import XAIChem


@lru_cache(maxsize=None)
def createMc(N, t):
    """
    Creat a matrix representation of an associated game as defined by Hamiache
    (https://doi.org/10.1007/s00182-019-00688-y).

    :param N: a set of players
    :param t: hyperparameter controlling the propotion of surplus
    """

    M = np.zeros(shape=(2 ** len(N) - 1, 2 ** len(N) - 1))
    all_coalitions = list(XAIChem.graph.powerset(N))

    # Diagonal elements
    for i, S in enumerate(all_coalitions):
        M[i, i] = 1 - (len(N) - len(S)) * t

        for j, T in enumerate(all_coalitions):
            if len(T) == 1 and not set(T).issubset(set(S)):
                M[i, j] = -t

            elif len(S) + 1 == len(T) and set(S).issubset(set(T)):
                M[i, j] = t

    return M


def createPg(N: frozenset | tuple, g: Iterable) -> np.ndarray:
    """
    Create the matrix Pg as defined by Hamiache containing the graphical
    information. (https://doi.org/10.1007/s00182-019-00688-y)

    :param N: a set of players
    :param g: a set of adjacent players
    """

    P = np.zeros(shape=(2 ** len(N) - 1, 2 ** len(N) - 1))
    all_coalitions = list(XAIChem.graph.powerset(N))

    for i, S in enumerate(all_coalitions):
        g_S = XAIChem.graph.inducedGraph(S, g)
        for R in XAIChem.graph.partition(S, g_S):
            P[i, all_coalitions.index(R)] = 1

    return P


def hamiacheNavarroValue(
    N: frozenset | tuple, v: np.ndarray, g: Iterable, t: float | None = None
) -> np.ndarray:
    """
    Computes a value for a transfer utility game (TU game) with incomplete
    communication as developed by Hamiache and Navarro.

    :param N: set of players
    :param v: characteristic function mapping a coalition (i.e. a subset of N)
        to a real value, represented as a numpy array.
    :param g: set of adjacent players
    :param t: hyperparameter specifying the amount of surplus. (default is
        None)
    """

    Mc = createMc(N, t)
    Pg = createPg(N, g)

    Mg = Pg @ Mc @ Pg
    print(Mg)

    # Converge power series
    Mg_old = np.inf
    while not np.allclose(Mg, Mg_old):
        Mg_old = Mg
        Mg = Mg @ Mg

    print(Mg)

    # Return the value
    return Mg @ v
