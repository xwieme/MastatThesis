from itertools import chain, combinations
from typing import Iterable
from collections import deque

import numpy as np


def powerset(A: Iterable):
    """
    Generate the powerset of A, i.e. the set of all subsets. The empty set is 
    excluded.

    Example: powerset([1, 2, 3]) results in 
    [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """

    A = list(A)
    return chain.from_iterable(combinations(A, r) for r in range(1, len(A)+1))

def adjacencyMatrix(N, g):

    A = np.zeros((max(N) + 1, max(N) + 1))

    # Iterate throug every edge, set an element to one if an edge exists 
    # between its corresponding indices
    for i, j in g:
        A[i, j] = 1
        A[j, i] = 1

    return A


def path(S: Iterable, g_S, i, j) -> bool:
    """
    Check if there exists a path from i to j in the graph <S, g(S)>.

    :param S: a set of vertices
    :param g_S: set of adjacent vertices of the set S
    :param i: a vertex index
    :param j: a vertex index 
    """

    # Assume vertices are connected to themself
    if i == j: 
        return True 

    # If one of the vertices is not in the set S, then they are not connected
    if i not in S or j not in S:
        return False

    A = adjacencyMatrix(S, g_S)

    # Get all vertices adjacent to vertex i in the graph <S, g(S)>
    to_visit = deque(*np.where(A[i, :] == 1))
    visited = {i}

    # If no vertices are adjacent to vertex i, return False
    if len(to_visit) == 0:
        return False

    # Check every adjacent vertex recursivly until vertex j is found or all
    # connected vertices are checked.
    current_vertex = np.nan 
    while current_vertex != j and len(to_visit) != 0:
        current_vertex = to_visit.pop()
        # Mark current vertex as visited
        visited.add(current_vertex)

        # Get all connected vertices to the current vertex. If they are not 
        # visited yet, assign to to_visit
        for neighbor in np.where(A[current_vertex, :] == 1)[0]:
            if neighbor not in visited:
                to_visit.append(neighbor)

    return current_vertex == j


def inducedGraph(S: Iterable, g: Iterable) -> set:
    """
    Generate the induced graph of S on g, where S is a subset of the vertices 
    of the whole graph.

    :param S: set of vertices
    :param g: set of adjacent vertices in the whole graph
    """
    return {(i, j) for i, j in g if i in S and j in S}


def partition(S: Iterable, g_S: Iterable):
    """
    Create a set of connected vertices of the graph <S, g(S)>. If the graph 
    does not contain isolated vertices, the result is equal to the set S.

    :param S: set of vertices 
    :param g_S: set of adjacent vertices in S
    """

    out = [] 
    S_copy = set(S)

    while len(S_copy) != 0:

        # Select initial vertex
        j = S_copy.pop()
        component = {j}

        # Get all vertices connected to j 
        for i in S_copy:
            if path(S, g_S, i, j):
                component.add(i)

        out.append(tuple(component))
        
        # Remove elements of the current component from S
        S_copy = S_copy - component
    
    return set(out)
