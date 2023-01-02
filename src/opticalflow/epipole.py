import numpy as np

class EpipoleComputation( object ):

    @staticmethod
    def epipole( u, v, smin, thresh, num_iterations=1000 ):
        """
        Takes flow (u,v) with confidence smin and finds the epipole using only the points with confidence above the threshold thresh 
        (for both sampling and finding inliers)
        params:
            @u: np.array(h,w)
            @v: np.array(h,w)
            @smin: np.array(h,w)
        return value:
            @best_ep: np.array(3,)
            @inliers: np.array(n,) 
        
        u, v and smin are (h,w), thresh is a scalar
        output should be best_ep and inliers, which have shapes, respectively (3,) and (n,) 
        """
            
        K = np.linspace( -256, 255, 512 )
        x_p, y_p = np.meshgrid( K, K )

        flattened_smin = smin.flatten()
        thresholded_x_p = x_p.flatten()[flattened_smin > thresh]
        thresholded_y_p = y_p.flatten()[flattened_smin > thresh]
        xp = np.vstack( (thresholded_x_p, thresholded_y_p, np.ones( len( thresholded_x_p ) ) ) )

        thresholded_u = u.flatten()[flattened_smin > thresh]
        thresholded_v = v.flatten()[flattened_smin > thresh]
        up = np.vstack( (thresholded_u, thresholded_v, np.zeros( len( thresholded_u ) ) ) )

        val = np.cross( xp.T, up.T )

        sample_size = 2

        eps = 10**-2

        best_num_inliers = -1
        best_inliers = None
        best_ep = None

        indices_above_threshold = np.where( smin.flatten() > thresh)[0]
        for i in range(num_iterations):
            permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(0,np.sum((smin>thresh))))
            sample_indices = permuted_indices[:sample_size]
            test_indices = permuted_indices[sample_size:]

            e = val[sample_indices]

            U, S, Vt = np.linalg.svd( e )
            ep = Vt[-1, :]

            inliers = indices_above_threshold[sample_indices]

            e_test = val[test_indices]
            dist_fc = np.abs( e_test @ ep )

            distances_bw_limit = test_indices[ np.where( dist_fc < eps )[0] ]
            inliers_from_dist_fc = indices_above_threshold[ distances_bw_limit ]
            inliers = np.append( inliers, inliers_from_dist_fc )
            
            if inliers.shape[0] > best_num_inliers:
                best_num_inliers = inliers.shape[0]
                best_ep = ep
                best_inliers = inliers

        return best_ep, best_inliers