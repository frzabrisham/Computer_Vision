
# In[ ]:


import numpy as np
from cv2 import *
from numpy.linalg import inv

folder = 'Frames/'
path450 = folder + 'f0450.jpg'
path270 = folder + 'f0270.jpg'
path630 = folder + 'f0630.jpg'
path90 = folder + 'f0090.jpg'
path810 = folder + 'f0810.jpg'
path1 = folder + 'f0001.jpg'

img450, im450 = imread(path450), imread(path450)
img270 = imread(path270)
img630 = imread(path630)
img90 = imread(path90)
img810 = imread(path810)
img1 = imread(path1)

height = int(img450.shape[0])
width = int(img450.shape[1])

image_folder = os.fsencode(folder)
filenames = []
for file in os.listdir(image_folder):
    filename = os.fsdecode(file)
    if filename.endswith('.jpg'):
        filenames.append(filename)

filenames.sort()


# In[ ]:



sift = cv2.SIFT_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = FlannBasedMatcher(index_params, search_params)

kp450, des450 = sift.detectAndCompute(img450, None)
kp270, des270 = sift.detectAndCompute(img270, None)
kp630, des630 = sift.detectAndCompute(img630, None)
kp90, des90 = sift.detectAndCompute(img90, None)
kp810, des810 = sift.detectAndCompute(img810, None)
kp1, des1 = sift.detectAndCompute(img1, None)



# In[ ]:



def matching(matches, kp_main, kp_min):
    matched_main, matched = [], []
    thresh = 0.6
    for match in matches:
        p1 = kp_main[match[0].queryIdx].pt
        p2 = kp_min[match[0].trainIdx].pt

        if match[0].distance < thresh * match[1].distance:
            matched_main.append([p1])
            matched.append([p2])

    return matched_main, matched



# In[ ]:




def find_hom(des_ref, des_minor, kp_ref, kp_minor):
    matches = flann.knnMatch(des_ref, des_minor, k=2)
    matched_1, matched_2 = matching(matches, kp_ref, kp_minor)
    H, status = findHomography(np.array(matched_2), np.array(matched_1), RANSAC, maxIters=10000)
    return H


# In[ ]:



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

    return [int(x_cor[0][0]) + 100, int(y_cor[0][0]) + 100]


def trans(H):
    t0, t1, t2, t3 = corner(H)
    x_cor = min(min(min(t0[0], t1[0]), t2[0]), t3[0])
    y_cor = min(min(min(t0[1], t1[1]), t2[1]), t3[1])
    p = [x_cor[0][0], y_cor[0][0]]
    t = ([[1, 0, -p[0] + 200], [0, 1, -p[1] + 100], [0, 0, 1]])
    return t

 

# In[ ]:


def masking(im):
    isNotEmpty = False
    mask = np.zeros((im.shape[0], im.shape[1]))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j][0] != 0 or im[i][j][1] != 0 or im[i][j][2] != 0:
                mask[i][j] = 1
                isNotEmpty = True
    return mask, isNotEmpty


# In[ ]:


# im1 is under im2

def simple_merge(im1, im2):
    im = np.zeros((im1.shape[0], im1.shape[1], 3))
    mask1, _ = masking(im1)
    mask2, _ = masking(im2)
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            if mask1[i][j] == 1 and mask2[i][j] == 0:
                im[i][j] = im1[i][j]
            elif mask2[i][j] == 1:
                im[i][j] = im2[i][j]
    return im


# In[ ]:



def dp_merge(im1, im2):
    s0, s1 = im1.shape[0], im1.shape[1]
    im = np.zeros((s0, s1, 3))
    shared = np.zeros((s0, s1))
    mask1, _ = masking(im1)
    mask2, _ = masking(im2)
    for i in range(s0):
        for j in range(s1):
            shared[i, j] = 256
            if mask1[i, j] == 1 and mask2[i, j] == 0:
                im[i, j] = im1[i, j]
            elif mask2[i, j] == 1 and mask1[i, j] == 0:
                im[i, j] = im2[i, j]
            elif mask2[i, j] == 1 and mask1[i, j] == 1:
                a1, a2 = im1[i, j], im2[i, j]
                shared[i, j] = np.sqrt(np.sum((a1.astype('float64') - a2.astype('float64')) ** 2))
    path = find_path(shared, s0, s1)
    while len(path) > 0:
        p = path.pop()
        for j in range(s1 - 1):
            if j < p[1] and mask1[p[0], j] == 1:
                im[p[0], j] = im1[p[0], j]
            elif j >= p[1] and mask2[p[0], [j]] == 1:
                im[p[0], j] = im2[p[0], j]
    return im


def find_path(shared, s0, s1):
    dp = np.zeros((s0, s1))
    pth = np.zeros((s0, s1))
    for i in range(s0 - 1):
        for j in range(s1 - 1):
            mn = min(min(dp[i - 1, j - 1], dp[i - 1, j + 1]), dp[i - 1, j])[0][0]
            if mn == dp[i - 1, j - 1]:
                pth[i, j] = -1
            elif mn == dp[i - 1, j + 1]:
                pth[i, j] = 1
            elif mn == dp[i - 1, j]:
                pth[i, j] = 0

            dp[i, j] = shared[i, j] + mn
            if dp[i, j] == 0:
                dp[i, j] = 10000 + mn
            dp[i, 0] = 1000000 + mn
            dp[i, s1 - 1] = 1000000 + mn
    dp_min = 10000000000
    i, j = s0 - 2, s1 - 2

    for n in range(1, j):
        if dp_min > dp[i, n] != 0:
            dp_min = dp[i, n]
            j = n

    masir = list()
    while len(masir) < s0 - 2:
        j += int(pth[i, j])
        i -= 1
        masir.append([i, j])
    return masir



# In[ ]:



def correct_border(arr1, arr2):
    s0, s1 = arr1.shape[0], arr1.shape[1]
    im = np.zeros((s0, s1, 3))
    for i in range(s0):
        for j in range(s1):
            if arr2[i, j, 0] != 0 or arr2[i, j, 1] != 0 or arr2[i, j, 2] != 0:
                im[i, j] = arr1[i, j]
    return im



# In[ ]:




def writeIm(arr, name, fldr=''):
    imwrite(fldr + '/' + name + '.jpg', arr)



# In[ ]:



# find homography for key frames

h_450 = find_hom(des_ref=des450, des_minor=des450, kp_ref=kp450, kp_minor=kp450)

h_270 = find_hom(des_ref=des450, des_minor=des270, kp_ref=kp450, kp_minor=kp270)

h_630 = find_hom(des_ref=des450, des_minor=des630, kp_ref=kp450, kp_minor=kp630)

h_90 = find_hom(des_ref=des270, des_minor=des90, kp_ref=kp270, kp_minor=kp90)
h_90 = h_270 @ h_90

h_810 = find_hom(des_ref=des630, des_minor=des810, kp_ref=kp630, kp_minor=kp810)
h_810 = h_630 @ h_810

h_1 = find_hom(des_ref=des90, des_minor=des1, kp_ref=kp90, kp_minor=kp1)
h_1 = h_90 @ h_1



# In[ ]:



# warped rectangle

a, b = 1 / 3, 2 / 3
pts = np.array([a * width, a * height, a * width, b * height, b * width, b * height, b * width, a * height], np.int32)
pts = pts.reshape((-1, 1, 2))

color = (150, 200, 50)
thickness = 4

rect_450 = polylines(im450, [pts], isClosed=True, color=color, thickness=thickness)
writeIm(arr=rect_450, name='res01-450-rect')

H270_inv = inv(h_270)
rect = polylines(np.zeros((height, width, 3)), [pts], isClosed=True, color=color, thickness=thickness)
warped_rect = warpPerspective(rect, H270_inv, (width, height))

rect_270 = simple_merge(img270, warped_rect)
writeIm(arr=rect_270, name='res02-270-rect')



# In[ ]:



T_270 = trans(h_270)
H270 = T_270 @ h_270

H450 = T_270 @ h_450
d = dim(H450)
dim270_450 = (d[0], d[1])

warped_270 = warpPerspective(img270, H270, dim270_450)
warped_450 = warpPerspective(img450, H450, dim270_450)

panorama_270_450 = simple_merge(warped_270, warped_450)
writeIm(arr=panorama_270_450, name='res03-270-450-panorama')



# In[ ]:


T = trans(h_90)
H90 = T @ h_90
H270 = T @ h_270
H450 = T @ h_450
H630 = T @ h_630
H810 = T @ h_810
d = dim(H810)
dim_panorama = (d[0] + 200, d[1] + 100)

####
warped_90 = warpPerspective(img90, H90, dim_panorama)
warped_270 = warpPerspective(img270, H270, dim_panorama)
warped_450 = warpPerspective(img450, H450, dim_panorama)
warped_630 = warpPerspective(img630, H630, dim_panorama)
warped_810 = warpPerspective(img810, H810, dim_panorama)

panorama_90_270 = simple_merge(warped_90, warped_270)
panorama_270_450 = simple_merge(panorama_90_270, warped_450)
panorama_630_450 = simple_merge(panorama_270_450, warped_630)
panorama_0 = simple_merge(panorama_630_450, warped_810)

####
warped_90_1 = warpPerspective(img90, H90, dim_panorama, borderMode=cv2.BORDER_REPLICATE)
warped_270_1 = warpPerspective(img270, H270, dim_panorama, borderMode=cv2.BORDER_REPLICATE)
warped_450_1 = warpPerspective(img450, H450, dim_panorama, borderMode=cv2.BORDER_REPLICATE)
warped_630_1 = warpPerspective(img630, H630, dim_panorama, borderMode=cv2.BORDER_REPLICATE)
warped_810_1 = warpPerspective(img810, H810, dim_panorama, borderMode=cv2.BORDER_REPLICATE)

panorama_90_270_1 = dp_merge(warped_90_1, warped_270_1)
panorama_270_450_1 = dp_merge(panorama_90_270_1, warped_450_1)
panorama_630_450_1 = dp_merge(panorama_270_450_1, warped_630_1)
panorama_1 = dp_merge(panorama_630_450_1, warped_810_1)

####
panorama = correct_border(panorama_1, panorama_0)

writeIm(arr=panorama, name='res04-key-frames-panorama')


# In[ ]:



T_panorama = trans(h_1)
dim_ref_pln = [0, 0]
homos = list()
Homs = list()
for n in range(901):
    filename = folder + filenames[n]
    img = imread(filename)
    kp, des = sift.detectAndCompute(img, None)
    h_ref = None
    d_ref, k_ref = None, None
    if n <= 180:
        d_ref, k_ref = des90, kp90
        h_ref = h_90
    elif 180 < n <= 360:
        d_ref, k_ref = des270, kp270
        h_ref = h_270
    elif 360 < n <= 540:
        d_ref, k_ref = des450, kp450
        h_ref = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    elif 540 < n <= 720:
        d_ref, k_ref = des630, kp630
        h_ref = h_630
    else:
        d_ref, k_ref = des810, kp810
        h_ref = h_810
    hom = find_hom(des_ref=d_ref, des_minor=des, kp_ref=k_ref, kp_minor=kp)
    hom = h_ref @ hom
    homos.append(hom)
    Hom = T_panorama @ hom
    Homs.append(Hom)
    file_hom = open("Homography.txt", "a")
    file_hom.write('\'' + 'homography' + str(n) + ':' + '\'' + str(hom))

dim_ref_pln = dim(Homs.pop())
# l = len(Homs)

for m in range(900):
    n = 900 - m
    H = Homs.pop()
    filename = folder + filenames[n]
    img = imread(filename)
    warped = warpPerspective(img, H, (dim_ref_pln[0], dim_ref_pln[1]))

    writeIm(arr=warped, name='img0' + str(n), fldr='reference-plane')



# In[ ]:



homography_list = []
homography_inv_list = []

file_content = ''

with open('hom.txt') as file:
    file_content = file.read()


def convert_line_to_numbers(line_of_file):
    line_of_file = line_of_file.replace('[', '').replace(']', '')
    numbers = []
    for number in line_of_file.split(' '):
        try:
            numbers.append(float(number))
        except:
            continue
    return numbers


line_by_line = file_content.split('\n')
h = []
line1, line2, line3 = convert_line_to_numbers(line_by_line[0]), convert_line_to_numbers(
    line_by_line[1]), convert_line_to_numbers(line_by_line[2])
h.append(line1)
h.append(line2)
h.append(line3)
M = (np.array(h, dtype=np.float32))
T = trans(M)
for a in range(0, len(line_by_line) - 1, 3):
    h = []
    line1 = convert_line_to_numbers(line_by_line[a])
    line2 = convert_line_to_numbers(line_by_line[a + 1])
    line3 = convert_line_to_numbers(line_by_line[a + 2])
    h.append(line1)
    h.append(line2)
    h.append(line3)
    M = np.array(h, dtype=np.float32)
    homography_list.append(M)

for h in homography_list:
    H = T @ h
    h_inv = np.linalg.inv(H)
    homography_inv_list.append(h_inv)


# In[ ]:


folder = 'reference-plane/img0'
im = imread(folder + str(1) + '.jpg')
dim_bg = [im.shape[0], im.shape[1]]
v, u = 5, 5
panorama_bg = np.zeros((dim_bg[0],dim_bg[1],3))
x, y = int(dim_bg[0] / v), int(dim_bg[1] / u)
for i in range(v + 1):
    for j in range(u + 1):
        parts = []
        for n in range(1, 900):
            # img = imread(folder + filenames[n])
            # M = homography_list[n]
            # warp = warpPerspective(img, M, dim_bg)
            # part = warp[x * i:x * (i + 1), y * j:y * (j + 1), :].copy()
            img = imread(folder + str(n) + '.jpg')
            part = img[x * i:x * (i + 1), y * j:y * (j + 1), :].copy()

            if i == v and j == u:
                part = img[x * i:, :, :].copy()
            elif j == u and i == 0:
                part = img[:, j * y:, :].copy()

            parts.append(np.ma.array(part, mask=(part == 0)))
            # writeIm(part, 'part' + str(n), 'parts')
        a = np.ma.median(np.ma.array(parts), axis=0).astype('uint8')
        if i == v and j == u:
            panorama_bg[x * i:, :, :] = a.copy()
        elif j == u and i == 0:
            panorama_bg[:, j * y:, :] = a.copy()
        else:
            panorama_bg[x * i:x * (i + 1), y * j:y * (j + 1), :] = a.copy()

        writeIm(a, 'part' + str(i) + str(j), 'masks')
        writeIm(panorama_bg, 'pano' + str(i) + str(j), 'masks')

writeIm(panorama_bg, 'res06-background-panorama')



# In[ ]:


panorama_bg = imread('res06-background-panorama.jpg')
h_inv_last = homography_inv_list[len(homography_inv_list) - 1]
for n in range(len(homography_inv_list)):
    h_inv = homography_inv_list[n]

    img = warpPerspective(panorama_bg, h_inv, (width, height))
    writeIm(img, 'bg0' + str(n), 'background-frames')

    img2 = warpPerspective(panorama_bg, h_inv, (int(width * 1.5), height))
    writeIm(img2, 'bg0' + str(n), 'background-frames-wider')


# In[ ]:


thresh = 140

for i in range(900):
    print(i)
    im_main = imread(folder + filenames[i])
    im_bg = imread('background-frames/bg0' + str(i + 1) + '.jpg')
    s = 8
    k = (6 * s + 1, 6 * s + 1)

    diff = np.abs((im_main - im_bg))

    diff = diff[:, :, 0] + diff[:, :, 1] + diff[:, :, 2]
    diff = GaussianBlur(diff, ksize=k, sigmaX=s)

    mask = np.zeros((width, height, 3))
    mask[diff > thresh] = [0, 0, 255]

    l = 3
    kernel = np.ones((l, l), np.uint8)
    mask = morphologyEx(mask, MORPH_OPEN, kernel)

    im = im_main + mask * im_bg
    imwrite('parts/im0' + str(i) + '.jpg', im)


