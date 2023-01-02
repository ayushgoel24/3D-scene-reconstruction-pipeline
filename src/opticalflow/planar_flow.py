import numpy as np

class PlanarParameters( object ):

    @staticmethod
    def compute_planar_params(flow_x, flow_y, K,
                                up=[256, 0], down=[512, 256]):
        """
        params:
            @flow_x: np.array(h, w)
            @flow_y: np.array(h, w)
            @K: np.array(3, 3)
            @up: upper left index [i,j] of image region to consider.
            @down: lower right index [i,j] of image region to consider.
        return value:
            sol: np.array(8,)
        """

        K_inv = np.linalg.inv( K )
        K_homo = K / K[ -1, -1 ]

        u = flow_x / K_homo[ 0, 0 ]
        v = flow_y / K_homo[ 1, 1 ]

        A = []
        b = []
        for i in range( up[0], down[0] ):
            for j in range( up[1], down[1] ):
                X_inv = K_inv @ np.array([ j, i, 1 ])
                x = X_inv[0] / X_inv[2]
                y = X_inv[1] / X_inv[2]
    
                A.append([ x**2, x*y, x, y, 1, 0, 0, 0 ])
                A.append([ x*y, y**2, 0, 0, 0, y, x, 1 ])
                b.append([ u[ i, j ] ])
                b.append([ v[ i, j ] ])
        
        sol, _, _, _ = np.linalg.lstsq( np.array(A), np.array(b), rcond=None )
        sol = sol.reshape((-1))
        
        return sol.flatten()