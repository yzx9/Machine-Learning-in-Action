import operator as op
import os

import numpy as np

import kNN


def imgToVector(filePath: str):
    k = 32
    vec = np.zeros(k*k)
    with open(file=filePath) as f:
        for i in range(k):
            line = f.readline()

            for j in range(k):
                vec[k*i + j] = int(line[j])

    (_, filename) = os.path.split(filePath)
    digit = int(filename[0])

    return vec, digit


def loadImgs(dirPath: str):
    filePaths = map(lambda file: dirPath + "/" + file, os.listdir(dirPath))
    digits = map(imgToVector, filePaths)
    data = np.array(list(map(op.itemgetter(0), digits)))
    labels = list(map(op.itemgetter(1), digits))
    return data, labels


trainingImgs: tuple[np.ndarray, list[int]] = None


def getTrainingImgs():
    global trainingImgs
    if (trainingImgs == None):
        trainingImgs = loadImgs("./data/Ch02/digits/trainingDigits")
    return trainingImgs


def classifyDigit(imgVec: np.ndarray):
    data, labels = getTrainingImgs()
    result = kNN.classify0(imgVec, data, labels, 3)
    return int(result)


def main():
    data, labels = loadImgs("./data/Ch02/digits/testDigits")

    countError = 0
    for i in range(data.shape[0]):
        classifier = classifyDigit(data[i])
        print("the classifier came back with: {}, the real answer is: {}".format(
            classifier, labels[i]))
        if (classifier != labels[i]):
            countError += 1

    print("the total number of errors is: {}".format(countError))
    print("the error rate is: {}".format(countError/data.shape[0]))


if (__name__ == "__main__"):
    main()
