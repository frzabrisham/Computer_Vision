# in[ ]:
import cv2
import numpy as np
from ipython_genutils.py3compat import xrange

img03 = cv2.imread('im03.jpg')
img04 = cv2.imread('im04.jpg')

sift = cv2.xfeatures2d.SIFT_create()

kp03, des03 = sift.detectAndCompute(img03, None)
kp04, des04 = sift.detectAndCompute(img04, None)

img3 = cv2.drawKeypoints(img03, kp03, img03, color=(0, 255, 0))
img4 = cv2.drawKeypoints(img04, kp04, img04, color=(0, 255, 0))
kp_merge = np.zeros((max(img03.shape[0], img04.shape[0]), img03.shape[1] + img04.shape[1], 3))
kp_merge[:img03.shape[0], :img03.shape[1], :] = img3
kp_merge[504:504 + img04.shape[0], img03.shape[1]:, :] = img4

cv2.imwrite('res13_corners.jpg', kp_merge)

# in[ ]:

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des03, des04, k=2)

matched_points3, matched_points4 = [], []

match_im3 = img3
match_im4 = img4

thresh = 0.75
for match in matches:
    p1 = kp03[match[0].queryIdx].pt
    p2 = kp04[match[0].trainIdx].pt
    if match[0].distance < 0.7 * match[1].distance:
        cv2.circle(match_im3, (int(p1[0]), int(p1[1])), radius=7, color=(255, 0, 0), thickness=3)
        cv2.circle(match_im4, (int(p2[0]), int(p2[1])), radius=7, color=(255, 0, 0), thickness=3)
        matched_points3.append([p1])
        matched_points4.append([p2])

    elif match[0].distance < thresh * match[1].distance:
        cv2.circle(match_im3, (int(p1[0]), int(p1[1])), radius=7, color=(255, 0, 100), thickness=3)
        cv2.circle(match_im4, (int(p2[0]), int(p2[1])), radius=7, color=(255, 0, 100), thickness=3)
        matched_points3.append([p1])
        matched_points4.append([p2])

merge_corres = np.zeros((max(img03.shape[0], img04.shape[0]), img03.shape[1] + img04.shape[1], 3))
merge_corres[:img03.shape[0], :img03.shape[1], :] = match_im3
merge_corres[504:504 + img04.shape[0], img03.shape[1]:, :] = match_im4
cv2.imwrite('res14_correspondences.jpg', merge_corres)

matchesMask = [[0, 0] for i in xrange(len(matches))]

for i, (match1, match2) in enumerate(matches):
    if match1.distance < thresh * match2.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(matchColor=(255, 0, 0),
                   singlePointColor=(0, 255, 0),
                   matchesMask=matchesMask,
                   flags=0)

matches_line = cv2.drawMatchesKnn(img03, kp03, img04, kp04, matches, None, **draw_params)
cv2.imwrite('res15_matches.jpg', matches_line)

matchesMask = [[0, 0] for i in xrange(len(matches))]
thresh1 = 0.65
for i, (match1, match2) in enumerate(matches):
    if match1.distance < thresh1 * match2.distance:
        matchesMask[i] = [1, 0]


matches_line = cv2.drawMatchesKnn(img03, kp03, img04, kp04, matches, None, **draw_params)
cv2.imwrite('res16.jpg', matches_line)

hom, status = cv2.findHomography(np.array(matched_points4), np.array(matched_points3), cv2.RANSAC, maxIters=10000)
print(hom)
# print(np.sum(status))
inlier_merge = np.zeros((max(img03.shape[0], img04.shape[0]), img03.shape[1] + img04.shape[1], 3))
inlier_merge[:img03.shape[0], :img03.shape[1], :] = img3
inlier_merge[504:504 + img04.shape[0], img03.shape[1]:, :] = img4

for i in range(len(status)):
    st = status[i][0]
    p3 = matched_points3[i][0]
    p4 = matched_points4[i][0]
    # print(p3)
    if st == 1:
        cv2.circle(inlier_merge, (int(p3[0]), int(p3[1])), radius=5, color=(0, 0, 255), thickness=5)
        cv2.circle(inlier_merge, (img03.shape[1] + int(p4[0]), 504 + int(p4[1])), radius=5, color=(0, 0, 255),
                   thickness=5)
        cv2.line(inlier_merge, (int(p3[0]), int(p3[1])),
                 (img03.shape[1] + int(p4[0]), 504 + int(p4[1])), color=(0, 0, 255), thickness=3)
    else:
        cv2.circle(inlier_merge, (int(p3[0]), int(p3[1])), radius=5, color=(255, 0, 0), thickness=5)
        cv2.circle(inlier_merge, (img03.shape[1] + int(p4[0]), 504 + int(p4[1])), radius=5, color=(255, 0, 0),
                   thickness=5)
        cv2.line(inlier_merge, (int(p3[0]), int(p3[1])),
                 (img03.shape[1] + int(p4[0]), 504 + int(p4[1])), color=(255, 0, 0), thickness=3)

cv2.imwrite('res17.jpg', inlier_merge)

T = np.array([[1, 0, 2700], [0, 1, 100], [0, 0, 1]])
H = T @ hom
# height, width, channels = img03.shape
warped_im = cv2.warpPerspective(img04, H, (9000, 2500))

cv2.imwrite('res19.jpg', warped_im)
