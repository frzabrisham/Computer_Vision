import math

import numpy as np
from cv2 import *
from scipy.spatial.transform import Rotation

main_img = imread('vns.jpg')
img = main_img.copy()

vp_y, vp_x, vp_z = [-22388, 3709, 1], [9126, 2560, 1], [-3229, -129958, 1]
a1, a2, a3 = vp_y[0], vp_x[0], vp_z[0]
b1, b2, b3 = vp_y[1], vp_x[1], vp_z[1]

X = np.array([[a1 - a3], [b1 - b3], [a2 - a3], [b2 - b3]]).reshape((2, 2))
Y = np.array([[a2 * (a1 - a3) + b2 * (b1 - b3)], [a1 * (a2 - a3) + b1 * (b2 - b3)]]).reshape((2, 1))
P = np.linalg.inv(X) @ Y
px, py = P[0][0], P[1][0]

f2 = (-(px ** 2) - (py ** 2) + (a1 + a2) * px + (b1 + b2) * py - (a1 * a2 + b1 * b2))
f = math.sqrt(f2)

circle(img, (int(px), int(py)), 10, (10, 50, 200), 20)
loc = (int(img.shape[0] / 2), 100)
putText(img, 'f =13368.599052217636', loc, FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5, LINE_AA)
imwrite('res03.jpg', img)

K = np.array([[f, 0, px], [0, f, py], [0, 0, 1]]).reshape((3, 3))
K_inv = np.linalg.inv(K)
# print(K)

# print(vp_x)
B = np.transpose(K_inv) @ K_inv
y = math.sqrt(vp_y @ B @ np.transpose(vp_y))
x = math.sqrt(vp_x @ B @ np.transpose(vp_x))
z = math.sqrt(vp_z @ B @ np.transpose(vp_z))


def scalar_prod(a, arr):
    return [arr[0] * a, arr[1] * a, arr[2] * a]


vp_y = scalar_prod(1 / y, vp_y)
vp_x = scalar_prod(1 / x, vp_x)
vp_z = scalar_prod(1 / z, vp_z)

vp = np.transpose(np.array([vp_x, vp_y, vp_z]))
R = np.array(K_inv @ vp)

r1 = [0, -1, 0]
r2 = [0, 0, -1]
r3 = [1, 0, 0]
R_prim = np.array([r1, r2, r3]).reshape(3, 3)

M = R_prim @ np.linalg.inv(R)

print(Rotation.from_matrix(M).as_euler('ZYX', degrees=True))
print(Rotation.from_matrix(M).as_euler('ZYX', degrees=False))
