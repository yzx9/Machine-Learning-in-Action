import matplotlib.pyplot as plt
import numpy as np

import kNN


def fileToMatrix(file: str):
    with open(file=file) as f:
        rows = [l.strip().split('\t') for l in f.readlines()]
        classLabelVector = [int(row[-1]) for row in rows]
        dataRaw = np.array([list(map(float, row[0:3])) for row in rows])
        return dataRaw, classLabelVector


def plot():
    data, _ = fileToMatrix("./data/Ch02/datingTestSet2.txt")
    _, ax = plt.subplots()
    ax.scatter(data[:, 1], data[:, 2])
    plt.show()


def classifyPerson(percentTats: float, ffMiles: float, icecream: float):
    dataRaw, labels = fileToMatrix("./data/Ch02/datingTestSet2.txt")
    data, norm = kNN.autoNorm(dataRaw)
    inArray = norm(np.array([percentTats, ffMiles, icecream]))
    return kNN.classify0(inArray, data, labels, 3)


def main():
    resultList = ['not at all', 'in small doses', 'in large doses']

    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent filier miles earned per year?"))
    icecream = float(input("liters of ice cream consumed per year?"))

    result = classifyPerson(percentTats, ffMiles, icecream)
    print("You will probably like this person: ", resultList[result - 1])


if (__name__ == '__main__'):
    main()
