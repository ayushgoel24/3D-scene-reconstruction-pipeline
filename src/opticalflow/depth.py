import numpy as np

class DepthComputation( object ):

    @staticmethod
    def depth( flow, confidence, ep, K, thres=10 ):
        """
        params:
            @flow: np.array(h, w, 2)
            @confidence: np.array(h, w, 2)
            @K: np.array(3, 3)
            @ep: np.array(3,) the epipole found in epipole.py note it is uncalibrated and you need to calibrate it in this function!
        return value:
            depth_map: np.array(h, w)
        """

        depth_map = np.zeros_like(confidence)

        K_homogenized = K / K[ -1, -1 ]
        K_inv = np.linalg.inv( K_homogenized )
        flow_x = K_inv[ 0, 0 ]
        flow_y = K_inv[ 1, 1 ]

        u = flow[ :, :, 0 ] / flow_x
        v = flow[ :, :, 1 ] / flow_y

        K_inv_ep = K_inv @ ep
        ep_x = K_inv_ep[ 0 ]
        ep_y = K_inv_ep[ 1 ]

        for i in range( u.shape[0] ):
            for j in range( u.shape[1] ):

                c = u[ i, j ]
                d = v[ i, j ]
                X = K_inv @ np.array( [ j, i, 1 ] )

                a = X[0] / X[2] - ep_x
                b = X[1] / X[2] - ep_y

                if( confidence[i, j] > thres ):
                    depth_map[i][j] = np.sqrt( (a**2 + b**2) / (c**2 + d**2) )

        truncated_depth_map = np.maximum(depth_map, 0)
        valid_depths = truncated_depth_map[truncated_depth_map > 0]
        # change the depth bound for better visualization if your depth is in different scale
        depth_bound = valid_depths.mean() + 10 * np.std(valid_depths)

        truncated_depth_map[truncated_depth_map > depth_bound] = 0
        truncated_depth_map = truncated_depth_map / truncated_depth_map.max()
        

        return truncated_depth_map