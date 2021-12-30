import math
import random

import numpy as np
from cv2 import *
from matplotlib import pyplot as plt

img1 = imread('01.JPG')
img2 = imread('02.JPG')
shape = (img1.shape[0], img1.shape[1])

sift = xfeatures2d.SIFT_create()

kp01, des01 = sift.detectAndCompute(img1, None)
kp02, des02 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des01, des02, k=2)

matched_points1, matched_points2 = [], []

thresh = 0.75
for match in matches:
    p1 = kp01[match[0].queryIdx].pt
    p2 = kp02[match[0].trainIdx].pt
    if match[0].distance < thresh * match[1].distance:
        matched_points1.append([p1])
        matched_points2.append([p2])

pts1 = np.int32(matched_points1)
pts2 = np.int32(matched_points2)

F, mask = findFundamentalMat(pts1, pts2, FM_RANSAC)

print(F)

merge = np.zeros((shape[0], 2 * shape[1], 3))
merge[:, :shape[1], :] = img1
merge[:, shape[1]:, :] = img2

for i in range(len(mask)):
    st = mask[i][0]
    p1 = matched_points1[i][0]
    p2 = matched_points2[i][0]
    cl_inlier = (0, 255, 0)
    cl_outlier = (0, 0, 255)
    thickness = 3
    radius = 5
    # print(st)
    if st == 0:
        circle(merge, (int(p1[0]), int(p1[1])), radius=radius, color=cl_outlier, thickness=thickness)
        circle(merge, (img1.shape[1] + int(p2[0]), int(p2[1])), radius=radius, color=cl_outlier, thickness=thickness)

    else:
        circle(merge, (int(p1[0]), int(p1[1])), radius=radius, color=cl_inlier, thickness=thickness)
        circle(merge, (img1.shape[1] + int(p2[0]), int(p2[1])), radius=radius, color=cl_inlier, thickness=thickness)

imwrite('res05.jpg', merge)


def epP(A):
    _, _, V = np.linalg.svd(A)
    matrix = V[-1, :]
    return (matrix[0] / matrix[2]), (matrix[1] / matrix[2]), (matrix[2] / matrix[2])


e1 = epP(F)
e2 = epP(np.transpose(F))
print(e1)
print(e2)

plt.figure(num=None, figsize=(24, 18), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(img1[..., ::-1])
plt.axis("off")
plt.scatter(e1[0], e1[1], marker='o')
plt.savefig('res06.jpg', bbox_inches='tight')
plt.close()

plt.figure(num=None, figsize=(24, 18), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(img2[..., ::-1])
plt.axis("off")
plt.scatter(e2[0], e2[1], marker='o')
plt.savefig('res07.jpg', bbox_inches='tight')

pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

selected_pts = list()
colors = list()
k = 20

for j in range(k):
    selected_pts.append(random.randint(0, len(pts1)))
    colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))


def drawlines(im, lines, points):
    l, c = shape

    for y in range(k):
        x = selected_pts[y]
        color = colors[y]
        l, pt = lines[x], points[x][0]
        x0, y0 = map(int, [0, -l[2] / l[1]])
        x1, y1 = map(int, [c, -(l[2] + l[0] * c) / l[1]])
        line(im, (x0, y0), (x1, y1), color, 3)
        circle(im, tuple(pt), 3, color, 5)

    return im


def line_param(point1, point2):
    A = np.cross(point1, point2)
    n = math.sqrt(A[0] ** 2 + A[1] ** 2)
    return [A[0] / n, A[1] / n, A[2] / n]


def computeEpilines(pts, ep):
    lines = list()
    for p in pts:
        l = line_param((p[0][0], p[0][1], 1), ep)
        lines.append(l)

    return np.array(lines)


lines1 = computeEpilines(pts1, e1)
lines1 = lines1.reshape(-1, 3)
ep_im1 = drawlines(img1, lines1, pts1)

lines2 = computeEpilines(pts2, e2)
lines2 = lines2.reshape(-1, 3)
ep_im2 = drawlines(img2, lines2, pts2)

ep_merge = np.zeros((shape[0], 2 * shape[1], 3))
ep_merge[:, :shape[1], :] = ep_im1
ep_merge[:, shape[1]:, :] = ep_im2

imwrite('res08.jpg', ep_merge)
