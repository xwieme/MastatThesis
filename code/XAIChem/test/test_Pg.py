import numpy as np

import XAIChem


def test_Pg():
    """
    Example taken from: (https://doi.org/10.1007/s00182-019-00688-y)
    """

    # Graph: 0 --- 1 --- 2
    N = {0, 1, 2}
    g = {(0, 1), (1, 2)}

    Pg = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ]
    )

    assert np.array_equal(XAIChem.attribution._createPg(N, g), Pg)


if __name__ == "__main__":
    test_Pg()
