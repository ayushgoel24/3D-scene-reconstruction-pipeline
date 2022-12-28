import cv2
from environment import ApplicationProperties
import numpy as np
from twoviewSFM import Reprojections, Epipoles, PoseEstimation, Ransac, ThreeDReconstruction
# from ransac import Ransac
# from epipoles import Epipoles
# from pose import PoseEstimation
# from recon3d import ThreeDReconstruction
# from reprojection import Reprojections
import os

class ReconstructionFrom2DImages( object ):

    def __init__( self, applicationProperties: ApplicationProperties ) -> None:
        self.applicationProperties = applicationProperties
        self.outputDirectory = applicationProperties.get_property_value( "3DReconstructionFrom2D.outputDirectory" )

    def load_images( self ) -> list:
        left_image = cv2.imread( self.applicationProperties.get_property_value( "3DReconstructionFrom2D.leftImagePath" ) )
        right_image = cv2.imread( self.applicationProperties.get_property_value( "3DReconstructionFrom2D.rightImagePath" ) )
        images = [ left_image, right_image ]
        return images

    def detect_SIFT_features( self, images ) -> tuple:
        keypoints = []
        descriptions = []
        output_sift_image = []
        for im in images:
            gray = cv2.cvtColor( im, cv2.COLOR_RGB2GRAY )
            sift = cv2.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute( gray, None )
            keypoints.append( kp )
            descriptions.append( des )
            out_im = cv2.drawKeypoints( gray, kp, gray, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )
            output_sift_image.append( out_im )

        cv2.imwrite( os.path.join( self.outputDirectory, "sift_features_in_left_image.png" ), output_sift_image[0] )
        cv2.imwrite( os.path.join( self.outputDirectory, "sift_features_in_right_image.png" ), output_sift_image[1] )

        return ( keypoints, descriptions )

    def match_detected_keypoints( self, images, keypoints, descriptions ):
        bf = cv2.BFMatcher( crossCheck=True )
        matches = bf.match( descriptions[0], descriptions[1] )
        print("num matches", len(matches))
        matched_image = cv2.drawMatches( images[0][:, :, ::-1], keypoints[0], images[1][:, :, ::-1], keypoints[1], matches, None, flags=2)
        cv2.imwrite( os.path.join( self.outputDirectory, "sift_matches.png" ), matched_image )

        return matches

    def compute_calibrated_coordinates( self, matches, keypoints ) -> tuple:
        f1 = f2 = float( self.applicationProperties.get_property_value( "3DReconstructionFrom2D.focalLength" ) )
        u0 = float( self.applicationProperties.get_property_value( "3DReconstructionFrom2D.u0" ) )
        v0 = float( self.applicationProperties.get_property_value( "3DReconstructionFrom2D.v0" ) )

        K = np.array([[f1, 0, u0],
              [0, f2, v0],
              [0, 0, 1]])

        uncalibrated_1 = [[ keypoints[0][match.queryIdx].pt[0], keypoints[0][match.queryIdx].pt[1], 1 ] for match in matches ]
        uncalibrated_2 = [[ keypoints[1][match.trainIdx].pt[0], keypoints[1][match.trainIdx].pt[1], 1 ] for match in matches ]

        uncalibrated_1 = np.array( uncalibrated_1 ).T
        uncalibrated_2 = np.array( uncalibrated_2 ).T

        k_inv = np.linalg.inv( K )

        calibrated_1 = np.matmul( k_inv, uncalibrated_1 ).T
        calibrated_2 = np.matmul( k_inv, uncalibrated_2 ).T

        return ( K, uncalibrated_1, uncalibrated_2, calibrated_1, calibrated_2 )
        
    def estimate_ransac( self, images, matches, keypoints, calibrated_1, calibrated_2 ):
        E_ransac, inliers = Ransac.ransac_estimator( calibrated_1, calibrated_2 )
        print("E_ransac", E_ransac)
        print("Num inliers", inliers.shape)

        inlier_matches = [ matches[i] for i in inliers ]

        matched_image = cv2.drawMatches( images[0][:, :, ::-1],
                                        keypoints[0],
                                        images[1][:, :, ::-1],
                                        keypoints[1],
                                        inlier_matches, None, flags=2 )

        return ( E_ransac, inlier_matches )

    def plot_epipoles( self, images, inlier_matches, keypoints, E_ransac, K ):
        uncalibrated_inliers_1 = [[ keypoints[0][match.queryIdx].pt[0], keypoints[0][match.queryIdx].pt[1], 1 ] for match in inlier_matches ]
        uncalibrated_inliers_2 = [[ keypoints[1][match.trainIdx].pt[0], keypoints[1][match.trainIdx].pt[1], 1 ] for match in inlier_matches ]
        uncalibrated_inliers_1 = np.array( uncalibrated_inliers_1 ).T
        uncalibrated_inliers_2 = np.array( uncalibrated_inliers_2 ).T

        Epipoles.plot_epipolar_lines( images[0], images[1], uncalibrated_inliers_1, uncalibrated_inliers_2, E_ransac, K )

        return ( uncalibrated_inliers_1, uncalibrated_inliers_2 )

    def estimate_pose( self, E_ransac ):
        transform_candidates = PoseEstimation.pose_candidates_from_E( E_ransac )
        print("transform_candidates", transform_candidates)
        return transform_candidates

    def plot_reconstruction(P1, P2, T, R):
        # P1trans = (R @ P1.T).T + T

        # plt.figure(figsize=(6.4*2, 4.8*2))
        # ax = plt.axes()
        # ax.set_xlabel('x')
        # ax.set_ylabel('z')

        # for i in range(P1.shape[0]):
        #     plt.plot([0, P2[i, 0]], [0, P2[i, 2]], 'bs-')
        #     plt.plot([T[0], P1trans[i, 0]], [T[2], P1trans[i, 2]], 'ro-')
        # plt.plot([0], [0], 'bs')
        # plt.plot([T[0]], [T[2]], 'ro')
        pass

    def compute_reconstruction( self, transform_candidates, calibrated_1, calibrated_2 ) -> tuple:
        P1, P2, T, R = ThreeDReconstruction.reconstruct3D( transform_candidates, calibrated_1, calibrated_2 )
        self.plot_reconstruction( P1, P2, T, R )

        return ( P1, P2, T, R )

    def complete_pipeline( self ) -> None:
        images = self.load_images()
        
        keypoints, descriptions = self.detect_SIFT_features( images )
        
        matches = self.match_detected_keypoints( images, keypoints, descriptions )
        
        K, uncalibrated_1, uncalibrated_2, calibrated_1, calibrated_2 = self.compute_calibrated_coordinates( matches, keypoints )

        E_ransac, inlier_matches = self.estimate_ransac( images, matches, keypoints, calibrated_1, calibrated_2 )

        uncalibrated_inliers_1, uncalibrated_inliers_2 = self.plot_epipoles( images, inlier_matches, keypoints, E_ransac, K)

        transform_candidates = self.estimate_pose( E_ransac )

        P1, P2, T, R = self.compute_reconstruction( transform_candidates, calibrated_1, calibrated_2 )
        
        Reprojections.show_reprojections( images[0], images[1], uncalibrated_1, uncalibrated_2, P1, P2, K, T, R )