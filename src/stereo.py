import cv2
from environment import ApplicationProperties
import k3d
import matplotlib.pyplot as plt
import numpy as np
from src.multiviewstereo import DataLoader, TwoViewStereo

class Stereo( object ):

    def __init__( self, applicationProperties: ApplicationProperties ) -> None:
        self.applicationProperties = applicationProperties
        self.dataDirectory = applicationProperties.get_property_value( "Stereo.dataDirectory" )

    def load_dataset( self ) -> tuple:
        # reference: https://vision.middlebury.edu/mview/
        DATA = DataLoader.load_middlebury_data( self.dataDirectory )
        view_i, view_j = DATA[0], DATA[3]
        plt.subplot(1, 2, 1)
        plt.title("Sample: left view (reference)")
        plt.imshow(view_i["rgb"])
        plt.subplot(1, 2, 2)
        plt.title("Sample: right view")
        plt.imshow(view_j["rgb"])
        plt.show()

        return DATA, view_i, view_j

    def rectify_two_views( self, view_i, view_j ) -> tuple:
        R_wi, T_wi = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
        R_wj, T_wj = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj

        R_ji, T_ji, B = TwoViewStereo.compute_right2left_transformation( R_wi, T_wi, R_wj, T_wj )
        assert T_ji[1, 0] > 0, "here we assume view i should be on the left, not on the right"

        R_irect = TwoViewStereo.compute_rectification_R( T_ji )

        rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = TwoViewStereo.rectify_2view(
            view_i["rgb"],
            view_j["rgb"],
            R_irect,
            R_irect @ R_ji,
            view_i["K"],
            view_j["K"],
            u_padding=20,
            v_padding=20,
        )

        plt.subplot(2, 2, 1)
        plt.title("input view i")
        plt.imshow(cv2.rotate(view_i["rgb"], cv2.ROTATE_90_COUNTERCLOCKWISE))
        plt.subplot(2, 2, 2)
        plt.title("input view j")
        plt.imshow(cv2.rotate(view_j["rgb"], cv2.ROTATE_90_COUNTERCLOCKWISE))
        plt.subplot(2, 2, 3)
        plt.title("rect view i")
        plt.imshow(cv2.rotate(rgb_i_rect, cv2.ROTATE_90_COUNTERCLOCKWISE))
        plt.subplot(2, 2, 4)
        plt.title("rect view j")
        plt.imshow(cv2.rotate(rgb_j_rect, cv2.ROTATE_90_COUNTERCLOCKWISE))
        plt.tight_layout()
        plt.show()

        return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr, B, R_irect, R_wi, T_wi

    def compute_disparity( self, rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr ):

        assert K_i_corr[1, 1] == K_j_corr[1, 1], "same focal Y length"
        assert (K_i_corr[0] == K_j_corr[0]).all(), "same K on X dim"
        assert (rgb_i_rect.shape == rgb_j_rect.shape), "rectified two views to have the same shape"

        h, w = rgb_i_rect.shape[:2]

        d0 = K_j_corr[1, 2] - K_i_corr[1, 2]

        patches_i = TwoViewStereo.image2patch(rgb_i_rect.astype(float) / 255.0, 3)  # [h,w,k*k,3]
        patches_j = TwoViewStereo.image2patch(rgb_j_rect.astype(float) / 255.0, 3)  # [h,w,k*k,3]

        vi_idx, vj_idx = np.arange(h), np.arange(h)
        disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0
        valid_disp_mask = disp_candidates > 0.0

        # as an example
        u = 400

        buf_i, buf_j = patches_i[:, u], patches_j[:, u]
        value = TwoViewStereo.ssd_kernel(buf_i, buf_j)  # each row is one pix from left, col is the disparity

        _upper = value.max() + 1.0
        value[~valid_disp_mask] = _upper

        plt.subplot(1,2,1)
        # Viz the  disparity-cost of u=500, v=200 on left view
        v = 200
        plt.title("Cost-Disparity of one left pixel")
        plt.xlabel("Disparity")
        plt.ylabel("Cost")
        plt.plot(disp_candidates[v], value[v])
        plt.subplot(1,2,2)
        plt.title("The cost map of one left horizon col")
        plt.xlabel("Disparity")
        plt.ylabel("left pixel coordinates  v_L")
        plt.imshow(value)
        plt.tight_layout()
        plt.show()

        return value

    def compute_disparity_map( self, rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr ) -> tuple:
        disp_map, consistency_mask = TwoViewStereo.compute_disparity_map( rgb_i_rect, rgb_j_rect, d0=K_j_corr[1, 2] - K_i_corr[1, 2], k_size=5 )
        plt.imshow(cv2.rotate(consistency_mask, cv2.ROTATE_90_COUNTERCLOCKWISE))
        plt.title("L-R consistency check mask")
        plt.show()
        
        return disp_map, consistency_mask

    def compute_dep_and_pcl( self, disp_map, B, rgb_i_rect, K_i_corr ) -> tuple:
        dep_map, xyz_cam = TwoViewStereo.compute_dep_and_pcl(disp_map, B, K_i_corr)

        plt.subplot(1, 3, 1)
        plt.title("rgb")
        plt.imshow(cv2.rotate(rgb_i_rect, cv2.ROTATE_90_COUNTERCLOCKWISE))
        plt.subplot(1, 3, 2)
        plt.title("raw disparity")
        plt.imshow(cv2.rotate(disp_map, cv2.ROTATE_90_COUNTERCLOCKWISE))
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.title("raw depth")
        plt.imshow(cv2.rotate(dep_map, cv2.ROTATE_90_COUNTERCLOCKWISE))
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        return dep_map, xyz_cam

    def postprocessing( self, disp_map, dep_map, rgb_i_rect, xyz_cam, consistency_mask, R_irect, R_wi, T_wi ):
        mask, pcl_world, pcl_cam, pcl_color = TwoViewStereo.postprocess(
            dep_map,
            rgb_i_rect,
            xyz_cam,
            R_wc=R_irect @ R_wi,
            T_wc=R_irect @ T_wi,
            consistency_mask=consistency_mask,
            z_near=0.5,
            z_far=0.6,
        )

        mask = (mask > 0).astype(np.float)

        plt.subplot(1, 3, 1)
        plt.title("rgb")
        plt.imshow(cv2.rotate(rgb_i_rect, cv2.ROTATE_90_COUNTERCLOCKWISE))
        plt.subplot(1, 3, 2)
        plt.title("postproc disparity")
        plt.imshow(cv2.rotate(disp_map * mask, cv2.ROTATE_90_COUNTERCLOCKWISE))
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.title("postproc depth")
        plt.imshow(cv2.rotate(dep_map * mask, cv2.ROTATE_90_COUNTERCLOCKWISE))
        plt.colorbar()
        plt.show()

        return mask, pcl_world, pcl_cam, pcl_color

    def viz_3d_embedded( self, pcl, color ):
        plot = k3d.plot(camera_auto_fit=True)
        color = color.astype(np.uint8)
        color32 = (color[:, 0] * 256**2 + color[:, 1] * 256**1 + color[:, 2] * 256**0).astype(
            np.uint32
        )
        plot += k3d.points(pcl.astype(float), color32, point_size=0.001, shader="flat")
        plot.display()

    def aggregate( self, DATA ):

        pcl_list, pcl_color_list, disp_map_list, dep_map_list = [], [], [], []
        pairs = [(0, 2), (2, 4), (5, 7), (8, 10), (13, 15), (16, 18), (19, 21), (22, 24), (25, 27)]

        for pair in pairs:
            i, j = pair
            _pcl, _pcl_color, _disp_map, _dep_map = TwoViewStereo.two_view(DATA[i], DATA[j], 5, TwoViewStereo.sad_kernel)
            pcl_list.append(_pcl)
            pcl_color_list.append(_pcl_color)
            disp_map_list.append(_disp_map)
            dep_map_list.append(_dep_map)

        plot = k3d.plot(camera_auto_fit=True)
        for pcl, color in zip(pcl_list, pcl_color_list):
            color = color.astype(np.uint8)
            color32 = (color[:, 0] * 256**2 + color[:, 1] * 256**1 + color[:, 2] * 256**0).astype(
                np.uint32
            )
            plot += k3d.points(pcl.astype(float), color32, point_size=0.001, shader="flat")
        plot.display()

    def complete_pipeline( self ) -> None:

        EPS = 1e-8

        DATA, view_i, view_j = self.load_dataset()

        rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr, B, R_irect, R_wi, T_wi = self.rectify_two_views( view_i, view_j )

        value = self.compute_disparity( rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr )

        # example for the pixel (u=500,v=300) from the left view
        v = 300
        best_matched_right_pixel = value[v].argmin()
        best_matched_left_pixel = value[:,best_matched_right_pixel].argmin()
        print(v, best_matched_left_pixel)
        consistent_flag = best_matched_left_pixel == v
        print(consistent_flag)

        # example for the pixel (u=500,v=380) from the left view
        v = 380
        best_matched_right_pixel = value[v].argmin()
        best_matched_left_pixel = value[:,best_matched_right_pixel].argmin()
        print(v, best_matched_left_pixel)
        consistent_flag = best_matched_left_pixel == v
        print(consistent_flag)

        disp_map, consistency_mask = self.compute_disparity_map( rgb_i_rect, rgb_j_rect, d0=K_j_corr[1, 2] - K_i_corr[1, 2], k_size=5 )

        dep_map, xyz_cam = self.compute_dep_and_pcl( disp_map, B, rgb_i_rect, K_i_corr )

        mask, pcl_world, pcl_cam, pcl_color = self.postprocessing( disp_map, dep_map, rgb_i_rect, xyz_cam, consistency_mask, R_irect, R_wi, T_wi )

        # SSD Two view reconstruction results
        self.viz_3d_embedded( pcl_world, pcl_color.astype(np.uint8) )

        # SAD Two view reconstruction results
        pcl_sad, pcl_color_sad, disp_map_sad, dep_map_sad = TwoViewStereo.two_view(DATA[0], DATA[2], 5, TwoViewStereo.sad_kernel)
        self.viz_3d_embedded(pcl_sad, pcl_color_sad.astype(np.uint8))

        # ZNCC Two view reconstruction results
        pcl_zncc, pcl_color_zncc, disp_map_zncc, dep_map_zncc = TwoViewStereo.two_view(DATA[0], DATA[2], 5, TwoViewStereo.zncc_kernel)
        self.viz_3d_embedded(pcl_zncc, pcl_color_zncc.astype(np.uint8))

        self.aggregate( DATA )

