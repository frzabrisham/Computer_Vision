import math

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from cv2 import *

main_img = imread('vns.jpg')
img = main_img.copy()

width, height = main_img.shape[1], main_img.shape[0]

gray = cvtColor(main_img, COLOR_RGB2GRAY)
kernel = np.ones((12, 12), np.uint8)
opening = morphologyEx(gray, MORPH_OPEN, kernel)
edges = Canny(opening, 350, 450)

for i in range(int(width / 2)):
    for j in range(int(height / 2)):
        edges[i, j] = 0


def line_param(point1, point2):
    A = np.cross(point1, point2)
    n = math.sqrt(A[0] ** 2 + A[1] ** 2)
    return [A[0] / n, A[1] / n, A[2] / n]


linesP = HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=200)
lines_x, lines_y, lines_z = list(), list(), list()

for l in linesP:
    x1, y1, x2, y2 = l[0]
    m = (y2 - y1) / (x2 - x1)

    k = 10000
    pt1 = (x1 - k, int(-m * k) + y1)
    pt2 = (x2 + k, int(m * k) + y2)
    p1 = (x1, y1, 1)
    p2 = (x2, y2, 1)
    lin = line_param(p1, p2)
    if m < 0:
        line(img, pt1, pt2, (0, 0, 128), 7)
        lines_y.append(lin)
    elif m < 10:
        line(img, pt1, pt2, (0, 128, 0), 7)
        lines_x.append(lin)
    else:
        line(img, pt1, pt2, (128, 0, 0), 7)
        lines_z.append(lin)

lines_y = np.array(lines_y).reshape((-1, 3))
lines_x = np.array(lines_x).reshape((-1, 3))
lines_z = np.array(lines_z).reshape((-1, 3))


def vanishP(A):
    _, _, V = np.linalg.svd(A)
    matrix = V[-1, :]
    return int(matrix[0] / matrix[2]), int(matrix[1] / matrix[2]), int(matrix[2] / matrix[2])


vp_y = tuple(vanishP(lines_y))
vp_x = tuple(vanishP(lines_x))
vp_z = tuple(vanishP(lines_z))

print(vp_x, vp_y, vp_z)

h = line_param(vp_y, vp_x)
print(h)


def newline(p1, p2, color):
    pic = plt.gca()
    xmin, xmax = pic.get_xbound()

    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin, ymax = pic.get_ybound()
    else:
        ymax = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmax - p1[0])
        ymin = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmin - p1[0])

    line_to_draw = mlines.Line2D([xmin, xmax], [ymin, ymax], color=color)
    pic.add_line(line_to_draw)
    return line_to_draw


plt.figure(num=None, figsize=(24, 18), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(main_img[..., ::-1])
plt.axis("off")
newline(vp_y, vp_x, 'r')

width = len(main_img[1])
height = len(main_img[0])
plt.scatter(width, 0, marker='o')

plt.savefig('res01.jpg', bbox_inches='tight')
plt.close()

plt.figure(num=None, figsize=(24, 18), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(main_img[..., ::-1])
plt.axis("off")
plt.scatter(vp_y[0], vp_y[1], marker='o')
plt.scatter(vp_x[0], vp_x[1], marker='o')
plt.scatter(vp_z[0], vp_z[1], marker='o')
newline(vp_y, vp_x, 'r')

plt.savefig('res02.jpg', bbox_inches='tight')
