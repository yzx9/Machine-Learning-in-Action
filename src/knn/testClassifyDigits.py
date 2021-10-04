import numpy as np


def imgToVector(filename: str):
    vec = np.zeros((1, 32*32))
    with open(file=filename) as f:
        for i in range(32):
            line = f.readline()

            for j in range(32):
                vec[0, 32*i + j] = int(line[j])

    return vec
