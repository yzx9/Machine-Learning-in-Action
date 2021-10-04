import operator as op
import numpy as np


def classify0(inX: list, dataSet: np.ndarray, labels: list[str], k: int) -> str:
    dataSetSize = dataSet.shape[0]
    diffMat: np.ndarray = np.tile(inX, (dataSetSize, 1)) - dataSet

    # calc euclidean distance
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()

    classCount: dict[str, int] = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(),
                              key=op.itemgetter(1),
                              reverse=True)

    return sortedClassCount[0][0]


def autoNorm(data: np.ndarray):
    # Auto Normalize
    minVec: np.ndarray = data.min(0)
    maxVec: np.ndarray = data.max(0)
    rangeVec: np.ndarray = maxVec - minVec

    def norm(vec: np.ndarray) -> np.ndarray:
        return (vec - minVec) / rangeVec

    normed = np.array(list(map(norm, data)))
    return normed, norm
