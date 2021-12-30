from os import listdir

import numpy as np
import pandas as pd
import seaborn as sn
from cv2 import *
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

folders_name = ['Inside_City', 'Store', 'Mountain', 'Tall_Building', 'Open_Country', 'Highway', 'Kitchen', 'Coast',
                'Office', 'Bedroom', 'Suburb', 'Livingroom', 'Street', 'Industrial', 'Forest']

sift = SIFT_create()
descriptors = list()


def find_feature(path, type_of_data):
    im_list = listdir(path)
    D = list()
    for file in im_list:
        if file.endswith(".jpg"):
            im = imread(path + file)
            _, des = sift.detectAndCompute(im, None)
            if type_of_data == 'Train':
                for d in des:
                    descriptors.append(d)

            D.append(des)
    return D


def init_desc(type_of_data):
    folder = 'Data/' + type_of_data + '/'
    desc = list()
    for f in folders_name:
        desc.append(find_feature(folder + f + '/', type_of_data))
    return desc


desc_train = init_desc('Train')
desc_test = init_desc('Test')

n_cluster = [50, 75, 85, 100]

Train_X, Train_Y = list(), list()
Test_X, Test_Y, True_Y = list(), list(), list()
max_percent = 0
for n in n_cluster:
    k_means = KMeans(n_clusters=n, random_state=0).fit(descriptors)

    train_X, train_Y = list(), list()
    test_X, test_Y, true_Y = list(), list(), list()


    def training(index):
        for des in desc_train[index]:
            h = np.zeros(n)
            for d in des:
                p = k_means.predict([d])
                h[p] += 1
            train_X.append(h.reshape(-1))
            train_Y.append(index)


    for i in range(len(folders_name)):
        training(i)


    def test(index):
        for des in desc_test[index]:
            h = np.zeros(n)
            for d in des:
                p = k_means.predict([d])
                h[p] += 1
            test_X.append(h.reshape(-1))
            true_Y.append(index)


    for i in range(len(folders_name)):
        test(i)

    Train_X.append([n, train_X])
    Train_Y.append([n, train_Y])
    Test_X.append([n, test_X])
    Test_Y.append([n, test_Y])

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(train_X, train_Y)
    test_Y = clf.predict(test_X)

    C = confusion_matrix(true_Y, test_Y)
    correct = 0
    for i in range(len(folders_name)):
        correct += C[i, i]
    print('n_cluster: ' + str(n))
    per = correct / len(folders_name)
    print(per)
    if per > max_percent:
        max_percent = per
        df_cm = pd.DataFrame(C, folders_name, folders_name)
        sn.set(font_scale=0.4)
        sn.heatmap(df_cm, annot=True)

        plt.savefig('res09.jpg')
