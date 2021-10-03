import kNN

group, labels = kNN.createDataSet()
print(group, labels)

class0 = kNN.classify0([0, 0], group, labels, 3)
print(class0)
