import numpy as np
import kNN


def createDataSet() -> tuple[np.ndarray, list[str]]:
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables


group, labels = createDataSet()
print(group, labels)

class0 = kNN.classify0([0, 0], group, labels, 3)
print(class0)
