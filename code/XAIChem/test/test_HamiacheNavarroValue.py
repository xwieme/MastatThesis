import numpy as np 

import XAIChem


def test_hamiacheNavarroValue():
    """
    Examples taken from https://doi.org/10.1007/s00182-019-00688-y
    """

    # Example 1
    N = (0, 1, 2, 3)
    g = {(0, 2), (0, 1), (0, 3), (2, 1)}
    v = np.zeros(2**len(N) - 1)
    v[-1] = 1

    assert np.allclose(
        XAIChem.attribution.hamiacheNavarroValue(N, v, g, 2/len(N))[:len(N)],
        [4/9, 7/36, 7/36, 1/6]
    )

    # Example 2
    N = (0, 1, 2, 3)
    g = {(0, 2), (0, 1), (0, 3), (2, 1), (3, 1)}
    v = np.zeros(2**len(N) - 1)
    v[-1] = 1

    assert np.allclose(
        XAIChem.attribution.hamiacheNavarroValue(N, v, g, 2/len(N))[:len(N)],
        [5/18, 5/18, 2/9, 2/9]
    )

    # Example 3
    N = (0, 1, 2, 3, 4)
    g = {(4, 2), (2, 0), (2, 1), (0, 1), (0, 3), (1, 3)}
    v = np.zeros(2**len(N) - 1)
    v[-1] = 1

    assert np.allclose(
        XAIChem.attribution.hamiacheNavarroValue(N, v, g, 2/len(N))[:len(N)],
        [7/36, 7/36, 10/27, 7/54, 1/9]
    )

    # Example 4
    N = (0, 1, 2, 3, 4)
    g = {(2, 1), (2, 4), (2, 0), (1, 0), (1, 3), (0, 3), (0, 4), (3 ,4)}
    v = np.zeros(2**len(N) - 1)
    v[-1] = 1

    assert np.allclose(
        XAIChem.attribution.hamiacheNavarroValue(N, v, g, 2/len(N))[:len(N)],
        [3/14, 11/56, 11/56, 11/56, 11/56]
    )

    # Example 5
    N = (0, 1, 2, 3, 4)
    g = {(4, 1), (1, 0), (0, 2), (0, 3)}
    v = np.zeros(2**len(N) - 1)
    v[-1] = 1

    assert np.allclose(
        XAIChem.attribution.hamiacheNavarroValue(N, v, g, 2/len(N))[:len(N)],
        [5/12, 11/36, 7/72, 7/72, 1/12]
    )


if __name__ == "__main__":
    test_hamiacheNavarroValue()
