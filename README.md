# 3D Scene Reconstruction Pipeline

The entire pipeline is composed of various parts.

## 1) Using Optical Flow to get Point Correspondences and estimate Depths.

Optical flow is computed over a small window. The smallest singular value of spatiotemporal derivative matrix is calculated and the pixels above a certain threshold value are considered. 

### **Flow Vectors:** <br/>
![Alt text](output/opticalflow/flow_10.png)

### **Epipoles after RANSAC and satisfying planar condition equation by a threshold:**  <br/>
![Alt text](output/opticalflow/epipole_10.png)

### **Depths are then calculated by assuming pure translational motion:**  <br/>
![Alt text](output/opticalflow/depth_10.png)

## 2) Reconstruction of 3d scene from 2 views using 2 view sfm

We first identify important features using SIFT -
![Alt text](3D-reconstruction-from-2D-images/Results/SIFT-points.png)

We then match key points using both least square and RANSAC to prove effectiveness of ransac -
![Alt text](3D-reconstruction-from-2D-images/Results/Key-pts-using-lst-sq.png)
![Alt text](3D-reconstruction-from-2D-images/Results/Key-pts-using-RANSAC.png)

The resulting epipolar lines are as follows 
![Alt text](3D-reconstruction-from-2D-images/Results/Epipolar-lines.png)

Finally we reproject the points of one image onto the other
![Alt text](3D-reconstruction-from-2D-images/Results/Reprojection.png)

## 3) Lastly we recreate the 3D model from multi view sfm

Input views - 
![Alt text](Reconstruction-from-Multi-view-stereo/Results/Input-views.png)

Disparity -
![Alt text](Reconstruction-from-Multi-view-stereo/Results/Disparity.png)

Disparity and depth after post processing -
![Alt text](Reconstruction-from-Multi-view-stereo/Results/Postproc-Disparity-and-depth.png)

L-R Consistency check mask -
![Alt text](Reconstruction-from-Multi-view-stereo/Results/L-R-Consistency-Check-Mask.png)

Reconstructed 3d model from 2 views using ZNCC Kernel -
![Alt text](Reconstruction-from-Multi-view-stereo/Results/Reconstructed-3d-model-ZNCC.png)

Entire Reconstructed 3d model
![Alt text](Reconstruction-from-Multi-view-stereo/Results/Reconstructed-3d-model.png)
![Alt text](Reconstruction-from-Multi-view-stereo/Results/Reconstructed-3d-model2.png)
