import XAIChem 


def test_partition():

    # Graph: 0 --- 1 --- 2
    N = {0, 1, 2}
    g = {(0, 1), (1, 2)}

    assert XAIChem.graph.partition(N, g) == {(0, 1, 2)}
  
    # Graph: 0     2
    N = {0, 2}
    g = set()
    assert XAIChem.graph.partition(N, g) == {(0,), (2,)}
    
    # Graph: 0     2 --- 1
    N = {0, 1, 2}
    g = {(1, 2)}
    assert XAIChem.graph.partition(N, g) == {(0,), (1, 2,)}


if __name__ == "__main__":
    test_partition()
