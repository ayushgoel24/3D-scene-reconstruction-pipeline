import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np 
from environment import ApplicationProperties
from essential import EssentialMatrix
from ransac import Ransac
from epipoles import Epipoles

applicationProperties = ApplicationProperties("application.yml")
applicationProperties.initializeProperties()

## 3D Reconstruction from 2D Images ##

# load the images
left_image = cv2.imread( applicationProperties.get_property_value( "3DReconstructionFrom2D.leftImagePath" ) )
right_image = cv2.imread( applicationProperties.get_property_value( "3DReconstructionFrom2D.rightImagePath" ) )
images = [ left_image, right_image ]

# Detects SIFT features in all of the images
keypoints = []
descriptions = []
for im in images:
    gray = cv2.cvtColor( im, cv2.COLOR_RGB2GRAY )
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute( gray, None )
    keypoints.append( kp )
    descriptions.append( des )
    out_im = cv2.drawKeypoints( gray, kp, gray, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )

# Matches the detected keypoints between the images
bf = cv2.BFMatcher( crossCheck=True )
matches = bf.match( descriptions[0], descriptions[1] )
print("num matches", len(matches))
matched_image = cv2.drawMatches(images[0][:, :, ::-1], keypoints[0], images[1][:, :, ::-1], keypoints[1], matches, None, flags=2)
plt.figure(figsize=(6.4*2, 4.8*2))
plt.title("All Matches")
plt.imshow(matched_image)

# Compute calibrated coordinates
f1 = f2 = 552
u0 = 307.5
v0 = 205

K = np.array([[f1, 0, u0],
              [0, f2, v0],
              [0, 0, 1]])


uncalibrated_1 = [[keypoints[0][match.queryIdx].pt[0], keypoints[0][match.queryIdx].pt[1], 1] for match in matches]
uncalibrated_2 = [[keypoints[1][match.trainIdx].pt[0], keypoints[1][match.trainIdx].pt[1], 1] for match in matches]

uncalibrated_1 = np.array(uncalibrated_1).T
uncalibrated_2 = np.array(uncalibrated_2).T

k_inv = np.linalg.inv(K)

calibrated_1 = np.matmul(k_inv, uncalibrated_1).T
calibrated_2 = np.matmul(k_inv, uncalibrated_2).T

E_least = EssentialMatrix.compute_using_least_squares_estimation(calibrated_1, calibrated_2)

E_ransac, inliers = Ransac.ransac_estimator(calibrated_1, calibrated_2)
print("E_ransac", E_ransac)
print("Num inliers", inliers.shape)

inlier_matches = [matches[i] for i in inliers]

matched_image = cv2.drawMatches(images[0][:, :, ::-1],
                                keypoints[0],
                                images[1][:, :, ::-1],
                                keypoints[1],
                                inlier_matches, None, flags=2)
plt.figure(figsize=(6.4*2, 4.8*2))
plt.title("RANSAC Inlier Matches")
plt.imshow(matched_image)


uncalibrated_inliers_1 = [[keypoints[0][match.queryIdx].pt[0], keypoints[0][match.queryIdx].pt[1], 1] for match in inlier_matches]
uncalibrated_inliers_2 = [[keypoints[1][match.trainIdx].pt[0], keypoints[1][match.trainIdx].pt[1], 1] for match in inlier_matches]
uncalibrated_inliers_1 = np.array(uncalibrated_inliers_1).T
uncalibrated_inliers_2 = np.array(uncalibrated_inliers_2).T

Epipoles.plot_epipolar_lines(images[0], images[1], uncalibrated_inliers_1, uncalibrated_inliers_2, E_ransac, K)