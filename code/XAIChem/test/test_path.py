import XAIChem 


def test_path():

    N = {0, 1, 2}
    g = {(0, 1), (1, 2)}

    assert XAIChem.graph.path(N, g, 0, 1) == True 
    assert XAIChem.graph.path(N, g, 0, 2) == True 
    assert XAIChem.graph.path({0, 2}, {}, 0, 2) == False 


if __name__ == "__main__":
    test_path()
