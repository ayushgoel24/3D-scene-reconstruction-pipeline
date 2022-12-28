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

<br/>

## 2) Reconstruction of 3D scene from 2 views using 2-view SFM

### **Feature identification using SIFT**:
![Alt text](output/2viewSFM/sift_features_in_left_image.png)
![Alt text](output/2viewSFM/sift_features_in_right_image.png)

### **Key point matching using Least Squares and RANSAC**:
![Alt text](output/2viewSFM/lse_matches.png)
![Alt text](output/2viewSFM/ransac_matches.png)

### **Resulting Epipolar lines**:
![Alt text](output/2viewSFM/epipoles_left_image.png)
![Alt text](output/2viewSFM/epipoles_right_image.png)

### **Reprojection of points of one image onto the other**:
![Alt text](output/2viewSFM/reprojection_left_image.png)
![Alt text](output/2viewSFM/reprojection_right_image.png)

<br/>

## 3) Recreate 3D Model from multi-view SFM

### **Input views**: <br/>
![Alt text](output/multiviewSFM/input_images.png)

### **Disparity**: <br/>
![Alt text](output/multiviewSFM/disparity.png)

### **Disparity and depth after post processing**: <br/>
![Alt text](output/multiviewSFM/disparity_after_postprocessing.png)

### **L-R Consistency check mask**: <br/>
![Alt text](output/multiviewSFM/lr_consistency_check.png)

Reconstructed 3d model from 2 views using ZNCC Kernel -
![Alt text](output/multiviewSFM/zncc.png)

Entire Reconstructed 3d model
![Alt text](output/multiviewSFM/fullView.png)