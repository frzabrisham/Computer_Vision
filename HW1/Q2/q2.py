# import cmath as math
from cmath import *

import cv2
import numpy as np

im = cv2.imread('logo.png')
theta = atan(8 / 5)
px = im.shape[0] / 2
py = im.shape[1] / 2

si = sin(theta).real
co = cos(theta).real

k = np.array([[500, 0, px], [0, 500, py], [0, 0, 1]])
rot_x = np.array([[1, 0, 0], [0, co, -si], [0, si, co]])
rot_y = np.array([[co, 0, si], [0, 1, 0], [-si, 0, co]])
rot_z = np.array([[co, -si, 0], [si, co, 0], [0, 0, 1]])
rot = rot_y
c = np.array([[40], [0], [-25]])
t = np.dot(rot, -c)
# print(t)
cam = np.zeros((rot.shape[0], rot.shape[1] + t.shape[1]))
cam[:rot.shape[0], :rot.shape[1]] = rot
cam[rot.shape[0] - 3:, rot.shape[1]:] = t
# print(cam)
p = np.dot(k, cam)
# print(p)
p_prim = np.zeros((3, 3))
p_prim[:, :2] = p[:, :2]
p_prim[:, 2:] = p[:, 3:]
# print(p_prim)
p_inv = np.linalg.inv(p_prim)

# l : left  , r : right  , d : down  , u : up
ld = np.dot(p_inv, [[0], [0], [1]])
ld = ld / ld[2]
lu = np.dot(p_inv, [[0], [255], [1]])
lu = lu / lu[2]
rd = np.dot(p_inv, [[255], [0], [1]])
rd = rd / rd[2]
ru = np.dot(p_inv, [[255], [255], [1]])
ru = ru / ru[2]

final = np.zeros((500, 500, 3))

min_x, max_x = int(ld[0]), int(rd[0])
min_y, max_y = int(ld[1]), int(lu[1])

scale = 6
for i in range(scale * min_x, scale * max_x):
    for j in range(scale * min_y, scale * max_y):
        f = np.array([[i / scale], [j / scale], [0], [1]])
        point = (np.dot(p, f))
        m = 1 / point[2]
        x, y = int(m * point[0]), int(m * point[1])
        # print(x, y)
        s = 1
        a, b = 300, 200
        if 0 < x < 256 and 0 < y < 256:
            final[s * i + a, s * j + b, 0] = im[x, y, 0]
            final[s * i + a, s * j + b, 1] = im[x, y, 1]
            final[s * i + a, s * j + b, 2] = im[x, y, 2]

cv2.imwrite('res12.jpg', final)
final = np.zeros((500, 500, 3))

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# another try:
count = 0
for i in range(-100, 500):
    for j in range(-100, 500):
        f = np.array([[i], [j], [0], [1]])
        point = (np.dot(p, f))
        x, y = int((point[0]) / 400) - 10, int((point[1]) / 400) - 20
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        min_x, min_y = min(min_x, x), min(min_y, y)
        # print(x, y)
        # print(point[2])
        if 0 < x < im.shape[0] and 0 < y < im.shape[1]:
            final[i, j, 0] = im[x, y, 0]
            final[i, j, 1] = im[x, y, 1]
            final[i, j, 2] = im[x, y, 2]
            count += 1

cv2.imwrite('final1.jpg', final)

# failed
