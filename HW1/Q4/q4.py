import random

import cv2
import numpy as np

img03 = cv2.imread('im03.jpg')
img04 = cv2.imread('im04.jpg')

sift = cv2.xfeatures2d.SIFT_create()

kp03, des03 = sift.detectAndCompute(img03, None)
kp04, des04 = sift.detectAndCompute(img04, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des03, des04, k=2)

matched_points = []

thresh = 0.75
for match in matches:
    p1 = kp03[match[0].queryIdx].pt
    p2 = kp04[match[0].trainIdx].pt

    if match[0].distance < thresh * match[1].distance:
        matched_points.append([p1, p2])
        # matched_points4.append([p2])


# l = len(matched_points)

def compute_homo(P):
    n = len(P)
    h = np.zeros((2 * n, 9))
    # p0, q0, p1, q1, p2, q2, p3, q3 = P[0][0], P[0][1], P[1][0], P[1][1], P[2][0], P[2][1], P[3][0], P[3][1]
    j = 0
    matrix = None
    for i in range(n):
        x = P[i][0][0]
        y = P[i][0][1]
        u = P[i][1][0]
        v = P[i][1][1]

        h[j] = [-x, -y, -1, 0, 0, 0, v * x, v * y, v]
        h[j + 1] = [0, 0, 0, -x, -y, -1, u * x, u * y, u]
        j += 2
        U, S, V = np.linalg.svd(h, full_matrices=True)
        matrix = V[-1, :]
    return matrix.reshape((3, 3))


maxIter = 100000
it = 0
w = 0
thresh = 1
max_in = 0
best_homo = None
inlier_list = []
while it < maxIter and (max_in == 0 or it < np.log(1 - 0.99) / np.log(1 - (max_in / len(matched_points)) ** 4)):
    random_i = random.choice(matched_points)
    p0, q0 = random_i[0], random_i[1]
    random_i = random.choice(matched_points)
    p1, q1 = random_i[0], random_i[1]
    random_i = random.choice(matched_points)
    p2, q2 = random_i[0], random_i[1]
    random_i = random.choice(matched_points)
    p3, q3 = random_i[0], random_i[1]
    P = [[p0, q0], [p1, q1], [p2, q2], [p3, q3]]
    h = compute_homo(P)
    cnt = 0
    inlier = []
    for match in matched_points:
        p = match[0]
        q = match[1]
        a = h @ np.array([p[0], p[1], 1])
        a /= a[2]
        b = np.array([q[0], q[1], 1])
        if np.sqrt(np.sum((a.astype('float64') - b.astype('float64')) ** 2)) < thresh:
            cnt += 1
            inlier.append([p, q])
        if cnt > max_in:
            max_in = cnt
            best_homo = h
            inlier_list = inlier
        # print(cnt)
        it += 1

print(max_in)
h = compute_homo(inlier_list)
print(h)
T = np.array([[1, 0, 2700], [0, 1, 100], [0, 0, 1]])
H = T @ h
warped_im = cv2.warpPerspective(img04, H, (9000, 2500))

cv2.imwrite('res20.jpg', warped_im)
