from os import listdir

import numpy as np
from cv2 import *
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

folders_name = ['Inside_City', 'Store', 'Mountain', 'Tall_Building', 'Open_Country', 'Highway', 'Kitchen', 'Coast',
                'Office', 'Bedroom', 'Suburb', 'Livingroom', 'Street', 'Industrial', 'Forest']

for size in range(20, 30):
    train_X = []
    train_Y = []


    def training(path, index):
        for file in listdir(path):
            if file.endswith(".jpg"):
                im = imread(path + file)
                im = resize(im, (size, size))
                x = np.array(im).reshape((-1))
                train_X.append(x)
                train_Y.append(index)
            else:
                continue


    for i in range(len(folders_name)):
        filename = folders_name[i]
        training('Data/Train/' + filename + '/', i)

    test_X = []
    true_Y = []


    def test(path, index):
        for file in listdir(path):
            if file.endswith(".jpg"):
                im = imread(path + file)
                im = resize(im, (size, size))
                x = np.array(im).reshape((-1))
                test_X.append(x)
                true_Y.append(index)
            else:
                continue


    for i in range(len(folders_name)):
        filename = folders_name[i]
        test('Data/Test/' + filename + '/', i)

    for k in range(4):
        for l in range(2):
            metric = 'l' + str(l + 1)
            neigh = KNeighborsClassifier(n_neighbors=2 * k + 1, metric=metric)
            neigh.fit(train_X, train_Y)
            test_Y = neigh.predict(test_X)

            C = confusion_matrix(true_Y, test_Y)

            correct = 0
            for i in range(15):
                correct += C[i, i]
            print('size: ' + str(size), ' kNN: ' + str(2 * k + 1), ' metric: ' + metric)
            print(correct / 15)
