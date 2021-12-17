import cv2
import numpy as np
from cv2 import imwrite

im01 = cv2.imread('im01.jpg')
im02 = cv2.imread('im02.jpg')

arr01 = cv2.imread('im01.jpg', cv2.IMREAD_GRAYSCALE)
arr02 = cv2.imread('im02.jpg', cv2.IMREAD_GRAYSCALE)
merged_im = np.zeros((im01.shape[0], im01.shape[1] + im02.shape[1], 3))
merged_im[:, 0:im01.shape[1], :] = im01
merged_im[:, im01.shape[1]:, :] = im02

imwrite('merge.jpg', merged_im)

k = 0.001
threshold = 1e10
threshmatch = 0.85
size_filter = 9
sigma = 7
n = 23
kernel = np.ones((6, 6), np.uint8)

selected01 = np.zeros((arr01.shape[0], arr01.shape[1]))
selected02 = np.zeros((arr02.shape[0], arr02.shape[1]))

feature01 = dict()
feature02 = dict()


def normalize(arr):
    return 255 * (arr - arr.min()) / (arr.max() - arr.min())


def gauss(arr, size, sig):
    return cv2.GaussianBlur(arr, (size, size), sig)


# In[ ]:

img01 = gauss(arr01, 7, 1)
I01_x = cv2.Sobel(img01, cv2.CV_64F, 1, 0, ksize=5)
I01_y = cv2.Sobel(img01, cv2.CV_64F, 0, 1, ksize=5)

I01_x2 = I01_x * I01_x
I01_y2 = I01_y * I01_y
I01_xy = I01_x * I01_y

grad01 = (I01_x2 + I01_y2) ** 1 / 2

S01_x2 = gauss(I01_x2, size_filter, sigma)
S01_y2 = gauss(I01_y2, size_filter, sigma)
S01_xy = gauss(I01_xy, size_filter, sigma)

det01 = S01_x2 * S01_y2 - S01_xy ** 2
trace01 = S01_x2 + S01_y2

R01 = det01 - k * trace01 ** 2
thresh01 = R01.copy()

thresh01[thresh01 < threshold] = 0
thresh01[thresh01 > threshold] = 255

grad01 = normalize(grad01)
score01 = normalize(R01)

imwrite('res01_grad.jpg', grad01)
imwrite('res03_score.jpg', score01)
imwrite('res05_thresh.jpg', thresh01)

# In[ ]:

img02 = gauss(arr02, 5, 0)
I02_x = cv2.Sobel(img02, cv2.CV_64F, 1, 0, ksize=5)
I02_y = cv2.Sobel(img02, cv2.CV_64F, 0, 1, ksize=5)

I02_x2 = I02_x * I02_x
I02_y2 = I02_y * I02_y
I02_xy = I02_x * I02_y

grad02 = (I02_x2 + I02_y2) ** 1 / 2

S02_x2 = gauss(I02_x2, size_filter, sigma)
S02_y2 = gauss(I02_y2, size_filter, sigma)
S02_xy = gauss(I02_xy, size_filter, sigma)

det02 = S02_x2 * S02_y2 - S02_xy ** 2
trace02 = S02_x2 + S02_y2

R02 = det02 - k * trace02 ** 2

thresh02 = R02.copy()
thresh02[thresh02 < threshold] = 0
thresh02[thresh02 > threshold] = 255

grad02 = normalize(grad02)
score02 = normalize(R02)

imwrite('res02_grad.jpg', grad02)
imwrite('res04_score.jpg', score02)
imwrite('res06_thresh.jpg', thresh02)


# In[ ]:


def dfs(graph, x0, y0):
    path = set()
    stack = [(x0, y0)]
    while len(stack) > 0:
        (x, y) = stack.pop()
        source = (x, y)
        if source not in path and mark[x][y] == 0:
            path.add(source)
            mark[x][y] = 1
            if graph[x][y] == 0:
                continue

            for i in range(-1, 2):
                for j in range(-1, 2):
                    if graph[x + i][y + j] != 0:
                        stack.append((x + i, y + j))
    return path


def nms(i, path=None, arr=None):
    if arr is None:
        arr = []
    if path is None:
        path = set()
    max = 0
    n = (0, 0)
    if i == 1:
        for node in path:
            if R01[node[0]][node[1]] > max:
                max = R01[node[0]][node[1]]
                n = node
        # print(n[0], n[1])
        selected01[n[0]][n[1]] = 255
        feat(n[0], n[1], 1)
        cv2.circle(arr, (int(n[1]), int(n[0])), radius=10, color=(0, 255, 0), thickness=3)

    else:
        for node in path:
            if R02[node[0]][node[1]] > max:
                max = R02[node[0]][node[1]]
                n = node
        selected02[n[0]][n[1]] = 255
        feat(n[0], n[1], 2)
        cv2.circle(arr, (int(n[1]), int(n[0])), radius=10, color=(0, 255, 0), thickness=3)

    return arr


def feat(x, y, i):
    m = 11
    if i == 1:
        a = np.array(im01[x - m:x + m + 1, y - m:y + m + 1])
        b = a.reshape((1, 1587))
        feature01[(x, y)] = b
    elif i == 2:
        a = np.array(im02[x - m:x + m + 1, y - m:y + m + 1])
        b = a.reshape((1, 1587))
        feature02[(x, y)] = b


# In[ ]:


thresh01 = cv2.erode(thresh01, kernel, iterations=1)
mark = np.zeros((arr01.shape[0], arr01.shape[1]))
arr1 = cv2.imread('im01.jpg')

for i in range(arr01.shape[0]):
    for j in range(arr01.shape[1]):
        if thresh01[i][j] != 0 and mark[i][j] == 0:
            path = dfs(thresh01, i, j)
            arr1 = nms(1, path, arr1)
            # x += 1

imwrite('res07_harris.jpg', arr1)

thresh02 = cv2.erode(thresh02, kernel, iterations=1)
arr2 = cv2.imread('im02.jpg')
# implot1 = plt.imshow(arr2)
mark = np.zeros((arr02.shape[0], arr02.shape[1]))

for i in range(arr02.shape[0]):
    for j in range(arr02.shape[1]):
        if thresh02[i][j] != 0 and mark[i][j] == 0:
            path = dfs(thresh02, i, j)
            arr2 = nms(2, path, arr2)
            # x += 1

imwrite('res08_harris.jpg', arr2)

# In[ ]:

matching1 = dict()
matching2 = dict()

for i in feature01:
    q1 = None
    q2 = None
    d1 = 10000000.0
    d2 = 10000000.0
    m = 10000000.0
    # print(i)
    a1 = feature01.get(i)
    for j in feature02:
        a2 = feature02.get(j)
        if a1 is not None and a2 is not None:
            m = np.sqrt(np.sum((a1.astype('float64') - a2.astype('float64')) ** 2))

        if d1 >= m:
            d1, d2 = m, d1
            q1, q2 = j, q1

    if d1 / d2 < threshmatch:
        # print(d1, d2)
        matching1[i] = q1

for i in feature02:
    q1 = None
    q2 = None
    d1 = 10000000.0
    d2 = 10000000.0
    m = 10000000.0
    # print(i)
    a1 = feature02.get(i)
    for j in feature01:
        a2 = feature01.get(j)
        if a1 is not None and a2 is not None:
            m = np.sqrt(np.sum((a1.astype('float64') - a2.astype('float64')) ** 2))

        if d1 >= m:
            d1, d2 = m, d1
            q1, q2 = j, q1

    if d1 / d2 < threshmatch:
        # print(d1, d2)
        matching2[i] = q1

delet = dict()
for i in matching1:
    x = matching1.get(i)
    for j in matching1:
        y = matching1.get(j)
        if x == y and i != j:
            delet[i] = x
            delet[j] = y

for i in delet:
    matching1.pop(i)

delet = dict()
for i in matching2:
    x = matching2.get(i)
    for j in matching2:
        y = matching2.get(j)
        if x == y and i != j:
            delet[i] = x
            delet[j] = y

for i in delet:
    matching2.pop(i)

finalmatch = dict()

for i in matching1:
    for j in matching2:
        # print(i, j)
        if i == matching2.get(j):
            finalmatch[i] = j

final01 = im01
for i in finalmatch:
    cv2.circle(final01, (int(i[1]), int(i[0])), radius=10, color=(0, 255, 0), thickness=3)

imwrite('res09_corres.jpg', final01)

final02 = im02
for j in finalmatch.keys():
    i = finalmatch.get(j)
    cv2.circle(final02, (int(i[1]), int(i[0])), radius=10, color=(0, 255, 0), thickness=3)

imwrite('res10_corres.jpg', final02)

final = cv2.imread('merge.jpg')
merged_im = cv2.imread('merge.jpg')

for i in finalmatch:
    j = finalmatch.get(i)
    cv2.circle(merged_im, (int(i[1]), int(i[0])), radius=10, color=(120, 200, 150), thickness=4)
    cv2.circle(merged_im, (im01.shape[1] + j[1], j[0]), radius=10, color=(120, 200, 150), thickness=4)
    cv2.line(merged_im, (i[1], i[0]), (im01.shape[1] + j[1], j[0]), color=(0, 0, 255), thickness=2)

imwrite('res11.jpg', merged_im)
