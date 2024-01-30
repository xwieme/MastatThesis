import numpy as np

import XAIChem


def test_Mc():
    """
    Example taken from https://doi.org/10.1007/s00182-019-00688-y
    """
    N = (0, 1, 2)

    t = 1
    Mc = np.array([
        [1 - 2*t, -t, -t, t, t, 0, 0],
        [-t, 1 - 2*t, -t, t, 0, t, 0],
        [-t, -t, 1 - 2*t, 0, t, t, 0],
        [0, 0, -t, 1 - t, 0, 0, t],
        [0, -t, 0, 0, 1 - t, 0, t],
        [-t, 0, 0, 0, 0, 1 - t, t],
        [0, 0, 0, 0, 0, 0, 1]
    ])

    print(Mc)
    print(XAIChem.attribution.createMc(N, t))

    assert np.array_equal(XAIChem.attribution.createMc(N, t),  Mc)



if __name__ == "__main__":
    test_Mc()
