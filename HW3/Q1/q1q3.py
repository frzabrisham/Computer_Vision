import numpy as np
from cv2 import *

main_img = imread('vns.jpg')

height = main_img.shape[0]
width = main_img.shape[1]


def corner(H):
    t0 = H @ np.array([0, 0, 1])
    t0 = t0 / t0[2]
    t1 = H @ np.array([width - 1, 0, 1])
    t1 = t1 / t1[2]
    t2 = H @ np.array([0, height - 1, 1])
    t2 = t2 / t2[2]
    t3 = H @ np.array([width - 1, height - 1, 1])
    t3 = t3 / t3[2]
    return t0, t1, t2, t3


def dim(H):
    t0, t1, t2, t3 = corner(H)
    x_cor = max(max(max(t0[0], t1[0]), t2[0]), t3[0])
    y_cor = max(max(max(t0[1], t1[1]), t2[1]), t3[1])

    return int(x_cor[0][0]) + 100, int(y_cor[0][0]) + 100


def trans(H):
    t0, t1, t2, t3 = corner(H)
    x_cor = min(min(min(t0[0], t1[0]), t2[0]), t3[0])
    y_cor = min(min(min(t0[1], t1[1]), t2[1]), t3[1])
    p = [x_cor[0][0], y_cor[0][0]]
    t = ([[1, 0, -p[0] + 200], [0, 1, -p[1] + 100], [0, 0, 1]])
    return t


K = np.array([[1.33685991e+04, 0.00000000e+00, 1.56308100e+03],
              [0.00000000e+00, 1.33685991e+04, 1.47597803e+03],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
K_inv = np.linalg.inv(K)
M = np.array([[0.87030986, -0.08114127, -0.48577447],
              [0.036249, 0.99421315, -0.10112482],
              [0.49116876, 0.07040109, 0.8682148]])
Hom = K @ M @ K_inv
print(Hom)
T = trans(Hom)
Hom = (T @ Hom).astype(np.float32)
d = dim(Hom)

warp = warpPerspective(main_img, Hom, d)
imwrite('res04.jpg', warp)
