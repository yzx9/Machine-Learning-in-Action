import operator
import matplotlib.pyplot as plt
import numpy as np


def createDataSet() -> tuple[np.ndarray, list[str]]:
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables


def classify0(inX: list, dataSet: np.ndarray, labels: list[str], k: int) -> str:
    dataSetSize = dataSet.shape[0]
    diffMat: np.ndarray = np.tile(inX, (dataSetSize, 1)) - dataSet

    # euclidean distance
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()

    classCount: dict[str, int] = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)

    return sortedClassCount[0][0]


def fileToMatrix(file: str):
    with open(file=file) as f:
        rows = [l.strip().split('\t') for l in f.readlines()]
        classLabelVector = [int(row[-1]) for row in rows]
        returnMat = np.array([row[0:3] for row in rows])
        return returnMat, classLabelVector


data, labels = fileToMatrix("./data/Ch02/datingTestSet2.txt")
fig, ax = plt.subplots()
ax.scatter(data[:, 1], data[:, 2])
plt.show()
