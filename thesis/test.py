import numpy as np

np.set_printoptions(precision=2)

v = [0, 0, 0, 0, 0, 0, 1]
t = 0.01
new_v = np.zeros(len(v))

for _ in range(400):
    print(v)

    new_v[0] = v[0] + t * (v[3] - v[0] - v[1])
    new_v[1] = v[1] + t * (v[3] - v[0] - v[1] + v[4] - v[1] - v[2])
    new_v[2] = v[2] + t * (v[4] - v[2] - v[1])
    new_v[3] = v[3] + t * (v[6] - v[3] - v[2])
    new_v[4] = v[4] + t * (v[6] - v[4] - v[0])
    new_v[5] = new_v[0] + new_v[2]
    new_v[6] = v[6]

    v = new_v
