import random

import numpy as np
from cv2 import *
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import confusion_matrix, average_precision_score
from torch import tensor, float32
from torchvision.ops import nms

num_train = 5000
num_valid = 500
num_test = 500


def init_imList(src):
    l = list()
    source = os.fsencode(src)
    for img in os.listdir(source):
        path_im = src + "/" + os.fsdecode(img)
        if img is not None:
            l.append(path_im)
    return l


neg_imList = init_imList("NegData")
pos_imList = init_imList("PosData")

random.seed(100)
random.shuffle(neg_imList)

random.seed(100)
random.shuffle(pos_imList)


def feat_vec(hog, index, winSiz):
    winStride = winSiz
    padding = (0, 0)
    img_pos = np.uint8(imread(pos_imList[index]))
    img_pos = resize(img_pos, winSiz)
    h_pos = hog.compute(img_pos, winStride=winStride, padding=padding)

    img_neg = np.uint8(imread(neg_imList[index]))
    img_neg = resize(img_neg, winSiz)
    h_neg = hog.compute(img_neg, winStride=winStride, padding=padding)
    return np.array(h_pos).reshape(-1), np.array(h_neg).reshape(-1)


def training(hog, winSiz):
    train_X = []
    train_Y = []
    for i in range(num_train):
        h_p, h_n = feat_vec(hog=hog, index=i, winSiz=winSiz)

        train_X.append(h_p)
        train_Y.append([1])

        train_X.append(h_n)
        train_Y.append([-1])

    return np.array(train_X), np.array(train_Y)


def testing(hog, winSiz):
    test_X = []
    true_Y = []

    for i in range(num_train + 10, num_train + num_test + 10):
        h_p, h_n = feat_vec(hog=hog, index=i, winSiz=winSiz)

        test_X.append(h_p)
        true_Y.append([1])

        test_X.append(h_n)
        true_Y.append([-1])

    return np.array(test_X), np.array(true_Y)


def validation(hog, winSiz):
    valid_X = []
    valid_Y = []
    c = 100
    for i in range(num_train + num_test + c, num_train + num_test + num_valid + c):
        h_p, h_n = feat_vec(hog=hog, index=i, winSiz=winSiz)

        valid_X.append(h_p)
        valid_Y.append([1])

        valid_X.append(h_n)
        valid_Y.append([-1])

    return np.array(valid_X), np.array(valid_Y)


# list_sizes = [8, 16, 32, 64]
# list_sizes = [8, 16, 32]
max_per = 0
kernels = ["linear", "poly", "rbf"]
best_bl_size, best_bl_stride, best_cl_size, best_win, best_C, best_kernel = (16, 16), (8, 8), (8, 8), (32, 32), 3, "rbf"

for sc in range(1, 6):
    winSize = (int(sc * 16), int(sc * 16))
    block_size = (int(winSize[0] / 2), int(winSize[1] / 2))
    block_stride = (int(winSize[0] / 4), int(winSize[1] / 4))
    cell_size = block_stride
    H = HOGDescriptor(tuple(winSize), tuple(block_size), tuple(block_stride),
                      tuple(cell_size), 9)
    X_train, Y_train = training(H, winSiz=winSize)
    for kernel in kernels:
        C = random.randint(1, 3)
        sv = svm.SVC(C=C, kernel=kernel)
        sv.fit(X_train, Y_train.ravel())

        X_valid, Y_valid = validation(H, winSiz=winSize)
        Y_pred = sv.predict(X_valid)

        conf = confusion_matrix(Y_valid, Y_pred)
        num_true = conf[0, 0] + conf[1, 1]
        percent = (num_true / (2 * num_valid)) * 100

        if percent > max_per:
            max_per = percent
            best_bl_size, best_bl_stride, best_cl_size, best_win = [block_size, block_stride, cell_size, winSize]
            best_C, best_kernel = [C, kernel]

best_hog = HOGDescriptor((32, 32), (16, 16), (8, 8),
                         (8, 8), 9)
best_svm = svm.SVC(C=best_C, kernel=best_kernel, probability=True, cache_size=1000)

X_train, Y_train = training(best_hog, winSiz=best_win)
best_svm.fit(X_train, Y_train.ravel())

X_test, Y_test = testing(best_hog, winSiz=best_win)
Y_predict = best_svm.predict(X_test)

conf = confusion_matrix(Y_test, Y_predict)
num_true = conf[0, 0] + conf[1, 1]
percent = (num_true / (2 * num_test)) * 100

print("best winSize is :", best_win, "best cell is :", best_cl_size, ", best blockSize is :", best_bl_size,
      ", best blockStride is :", best_bl_stride, ", best C is :", best_C, ", best kernel is :", best_kernel, "   =>")
print("percent: ", percent)

metrics.plot_roc_curve(best_svm, X_test, Y_test)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig("res1.jpg")
plt.close()

metrics.plot_precision_recall_curve(best_svm, X=X_test, y=Y_test)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.savefig("res2.jpg")
plt.close()

average_precision = average_precision_score(Y_test, Y_predict)
print("AP:  ", average_precision)


def FaceDetector(main_img):
    boxes = []
    probs = []
    copy_img = main_img.copy()
    print(main_img.shape)
    winSiz = (128, 128)
    hog = best_hog

    for i in range(1, main_img.shape[0] - winSiz[0] - 1, 5):
        for j in range(1, main_img.shape[1] - winSiz[1] - 1, 5):
            im = main_img[i:i + winSiz[0], j:j + winSiz[1], :]
            h = hog.compute(np.uint8(im), winStride=(128, 128), padding=(1, 1))
            X_tst = [np.array(h).reshape(-1)]
            prob = best_svm.predict_proba(X_tst)[0, 1]

            if prob > 0.8:
                boxes.append([j, i, j + winSiz[1], i + winSiz[0]])
                probs.append(prob)

    indexes = nms(tensor(np.array(boxes), dtype=float32), tensor(probs, dtype=float32), 0.005)
    for index in indexes:
        pt0 = (boxes[index][0], boxes[index][1])
        pt1 = (boxes[index][2], boxes[index][3])
        rectangle(copy_img, pt0, pt1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)

    return copy_img


esteghlal = imread("Esteghlal.jpg")
persepolis = imread("Persepolis.jpg")
melli = imread("Melli.jpg")

pers = FaceDetector(persepolis)
imwrite("res5.jpg", pers)

mel = FaceDetector(melli)
imwrite("res4.jpg", mel)

est = FaceDetector(esteghlal)
imwrite("res6.jpg", est)
