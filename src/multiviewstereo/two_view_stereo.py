import cv2
import numpy as np

class TwoViewStereo( object ):

    @staticmethod
    def homo_corners( h, w, H ):
        corners_bef = np.float32( [[0, 0], [w, 0], [w, h], [0, h]] ).reshape( -1, 1, 2 )
        corners_aft = cv2.perspectiveTransform( corners_bef, H ).squeeze(1)
        u_min, v_min = corners_aft.min( axis=0 )
        u_max, v_max = corners_aft.max( axis=0 )
        return u_min, u_max, v_min, v_max

    @staticmethod
    def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):
        """Given the rectify rotation, compute the rectified view and corrected projection matrix

        Parameters
        ----------
        rgb_i,rgb_j : [H,W,3]
        R_irect,R_jrect : [3,3]
            p_rect_left = R_irect @ p_i
            p_rect_right = R_jrect @ p_j
        K_i,K_j : [3,3]
            original camera matrix
        u_padding,v_padding : int, optional
            padding the border to remove the blank space, by default 20

        Returns
        -------
        [H,W,3],[H,W,3],[3,3],[3,3]
            the rectified images
            the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
        """
        # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
        assert rgb_i.shape == rgb_j.shape
        
        h, w = rgb_i.shape[:2]

        ui_min, ui_max, vi_min, vi_max = TwoViewStereo.homo_corners( h, w, K_i @ R_irect @ np.linalg.inv(K_i) )
        uj_min, uj_max, vj_min, vj_max = TwoViewStereo.homo_corners( h, w, K_j @ R_jrect @ np.linalg.inv(K_j) )

        # The distortion on u direction (the world vertical direction) is minor, ignore this
        w_max = int( np.floor( max(ui_max, uj_max) ) ) - u_padding * 2
        h_max = int( np.floor( min(vi_max - vi_min, vj_max - vj_min) ) ) - v_padding * 2

        assert K_i[0, 2] == K_j[0, 2]
        K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
        K_i_corr[0, 2] -= u_padding
        K_i_corr[1, 2] -= vi_min + v_padding
        K_j_corr[0, 2] -= u_padding
        K_j_corr[1, 2] -= vj_min + v_padding

        rgb_i_rect = cv2.warpPerspective(rgb_i, ( K_i_corr @ R_irect @ ( np.linalg.inv( K_i ) ) ), dsize=( w_max, h_max ) )
        rgb_j_rect = cv2.warpPerspective(rgb_j, ( K_j_corr @ R_jrect @ ( np.linalg.inv( K_j ) ) ), dsize=( w_max, h_max ) )

        return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr

    @staticmethod
    def compute_right2left_transformation( R_wi, T_wi, R_wj, T_wj ):
        """Compute the transformation that transform the coordinate from j coordinate to i

        Parameters
        ----------
        R_wi, R_wj : [3,3]
        T_wi, T_wj : [3,1]
            p_i = R_wi @ p_w + T_wi
            p_j = R_wj @ p_w + T_wj
        Returns
        -------
        [3,3], [3,1], float
            p_i = R_ji @ p_j + T_ji, B is the baseline
        """

        R_ji = R_wi @ np.linalg.inv( R_wj )
        T_ji = -R_ji @ T_wj + T_wi
        B = np.linalg.norm( T_ji )

        return R_ji, T_ji, B

    @staticmethod
    def compute_rectification_R( EPS, T_ji ):
        """Compute the rectification Rotation

        Parameters
        ----------
        T_ji : [3,1]

        Returns
        -------
        [3,3]
            p_rect = R_irect @ p_i
        """
        # check the direction of epipole, should point to the positive direction of y axis
        e_i = T_ji.squeeze(-1) / ( T_ji.squeeze(-1)[1] + EPS )
        
        e_2 = ( T_ji / np.linalg.norm( T_ji + EPS ) ).flatten()
        
        e_1 = np.cross( e_2, np.array([0, 0, 1]) )
        e_1 = e_1 / np.linalg.norm( e_1 + EPS )

        e_3 = np.cross( e_1, e_2 )
        e_3 = e_3 /  np.linalg.norm( e_3 + EPS )

        R_irect = np.vstack((e_1, e_2, e_3))
        
        return R_irect

    @staticmethod
    def ssd_kernel(src, dst):
        """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

        Parameters
        ----------
        src : [M,K*K,3]
            M left view patches
        dst : [N,K*K,3]
            N right view patches

        Returns
        -------
        [M,N]
            error score for each left patches with all right patches.
        """

        assert src.ndim == 3 and dst.ndim == 3
        assert src.shape[1:] == dst.shape[1:]

        # Creating M * N * (K*K) * 3 matrix for each of these patches
        src = src[:, np.newaxis, :, :]
        dst = dst[np.newaxis, :, :, :]
        ssd = np.zeros( ( src.shape[0], dst.shape[1] ) )
        for i in range( 3 ):
            ssd += np.sum( np.square( src[:, :, :, i] - dst[:, :, :, i] ) , axis = 2 )

        return ssd

    @staticmethod
    def sad_kernel(src, dst):
        """Compute SAD Error, the RGB channels should be treated saperately and finally summed up

        Parameters
        ----------
        src : [M,K*K,3]
            M left view patches
        dst : [N,K*K,3]
            N right view patches

        Returns
        -------
        [M,N]
            error score for each left patches with all right patches.
        """

        assert src.ndim == 3 and dst.ndim == 3
        assert src.shape[1:] == dst.shape[1:]

        src = src[:, np.newaxis, :, :]
        dst = dst[np.newaxis, :, :, :]
        sad = np.zeros( ( src.shape[0], dst.shape[1] ) )
            
        for i in range( 3 ):
            sad += np.sum( np.abs( src[:, :, :, i] - dst[:, :, :, i] ) , axis = 2 )

        return sad

    @staticmethod
    def zncc_kernel(src, dst, EPS):
        """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

        Parameters
        ----------
        src : [M,K*K,3]
            M left view patches
        dst : [N,K*K,3]
            N right view patches

        Returns
        -------
        [M,N]
            score for each left patches with all right patches.
        """

        assert src.ndim == 3 and dst.ndim == 3
        assert src.shape[1:] == dst.shape[1:]

        zncc = np.zeros( ( src.shape[0], dst.shape[0] ) )
        for i in range( 3 ):
            src_mean = np.mean( src[:, :, i], axis = 1 ).reshape( (-1, 1) )
            src_sig = np.std( src[:, :, i], axis = 1 ).reshape( (-1, 1) )
            dst_mean = np.mean( dst[:, :, i], axis = 1 ).reshape( (-1, 1) )
            dst_sig = np.std( dst[:, :, i], axis = 1 ).reshape( (-1, 1) )

            w_1 = ( src[:, :, i] - src_mean )[:, np.newaxis, :]
            w_2 = ( dst[:, :, i] - dst_mean )[np.newaxis, :, :]
            
            zncc += np.sum( w_1 * w_2, axis=2 ) / ( (src_sig @ dst_sig.T) + EPS )

        # ! note here we use minus zncc since we use argmin outside, but the zncc is a similarity, which should be maximized
        return zncc * (-1.0)

    @staticmethod
    def image2patch(image, k_size):
        """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

        Parameters
        ----------
        image : [H,W,3]
        k_size : int, must be odd number; your function should work when k_size = 1

        Returns
        -------
        [H,W,k_size**2,3]
            The patch buffer for each pixel
        """

        padded_image = np.empty( ( image.shape[0] + k_size - 1, image.shape[1] + k_size - 1, image.shape[2] ) )
        for i in range(3):
            padded_image[:, :, i] = np.pad( image[:, :, i], int( k_size / 2 ), mode='constant' )
        patch = np.zeros( (image.shape[0], image.shape[1], k_size*k_size, 3) )

        for x in range( image.shape[1] ):
            for y in range( image.shape[0] ):
                index_y, index_x = np.meshgrid( np.arange( x - int( k_size / 2 ), x + int( k_size / 2 ) + 1 ), np.arange( y - int( k_size / 2 ), y + int( k_size / 2 ) + 1 ) )
                index_x += int( k_size / 2 )
                index_y += int( k_size / 2 )
                for i in range(3):
                    patch[y, x, :, i] = padded_image[ index_x, index_y, i ].flatten()

        return patch

    @staticmethod
    def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel, img2patch_func=image2patch):
        """Compute the disparity map from two rectified view

        Parameters
        ----------
        rgb_i,rgb_j : [H,W,3]
        d0 : see the hand out, the bias term of the disparty caused by different K matrix
        k_size : int, optional
            The patch size, by default 3
        kernel_func : function, optional
            the kernel used to compute the patch similarity, by default ssd_kernel
        img2patch_func : function, optional
            this is for auto-grader purpose, in grading, we will use our correct implementation of the image2path function to exclude double count for errors in image2patch function

        Returns
        -------
        disp_map: [H,W], dtype=np.float64
            The disparity map, the disparity is defined in the handout as d0 + vL - vR

        lr_consistency_mask: [H,W], dtype=np.float64
            For each pixel, 1.0 if LR consistent, otherwise 0.0
        """

        h, w = rgb_i.shape[:2]
        disp_map = np.empty( ( h, w ), dtype = np.float64 )
        lr_consistency_mask = np.zeros( ( h, w ), dtype = np.float64 )

        patches_i = img2patch_func( rgb_i.astype(float) / 255.0, k_size )
        patches_j = img2patch_func( rgb_j.astype(float) / 255.0, k_size )

        vi_idx, vj_idx = np.arange(h), np.arange(h)
        disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0

        for i in range(w):
            buf_i, buf_j = patches_i[:, i], patches_j[:, i]
            value = kernel_func( buf_i, buf_j )
            best_matched_right_pixel = np.argmin( value, axis = 1 )
            match = np.arange( h )
            disp_map[:, i] = disp_candidates[ match, best_matched_right_pixel ]
            best_matched_left_pixel = np.argmin( value[:, best_matched_right_pixel] , axis = 0 )
            consistent_flag = best_matched_left_pixel == vi_idx
            lr_consistency_mask[:, i] = consistent_flag

        return disp_map, lr_consistency_mask

    @staticmethod
    def compute_dep_and_pcl(disp_map, B, K):
        """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
        compute the depth map and backprojected point cloud

        Parameters
        ----------
        disp_map : [H,W]
            disparity map
        B : float
            baseline
        K : [3,3]
            camera matrix

        Returns
        -------
        [H,W]
            dep_map
        [H,W,3]
            each pixel is the xyz coordinate of the back projected point cloud in camera frame
        """

        dep_map = np.divide( K[1, 1] * B, disp_map )

        u, v = np.meshgrid( np.arange(disp_map.shape[1]), np.arange(disp_map.shape[0]) )
        xyz_cam = np.dstack( ( (u - K[0, 2]) * dep_map / K[0, 0], (v - K[1, 2]) * dep_map / K[1, 1], dep_map ) )

        return dep_map, xyz_cam