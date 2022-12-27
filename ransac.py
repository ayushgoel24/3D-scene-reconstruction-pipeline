import numpy as np
from essential import EssentialMatrix
import traceback

class Ransac( object ):

    def __init__(self) -> None:
        pass

    @staticmethod
    def ransac_estimator(X1, X2, num_iterations=60000):
        sample_size = 8

        eps = 10**-4

        best_num_inliers = -1
        best_inliers = None
        best_E = None

        try:

            for i in range(num_iterations):
                # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
                permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
                sample_indices = permuted_indices[:sample_size]
                test_indices = permuted_indices[sample_size:]

                E = EssentialMatrix.compute_using_least_squares_estimation( X1[sample_indices], X2[sample_indices] )
                
                Et = E.T

                inliers = []
                for k in test_indices:
                    d_x2_x1 = np.square( X2[k].T @ E @ X1[k] ) / ( np.linalg.norm( np.cross( [ 0, 0, 1 ] , E @ X1[k] ) ) ** 2 )
                    d_x1_x2 = np.square( X1[k].T @ Et @ X2[k] ) / ( np.linalg.norm( np.cross( [ 0, 0, 1 ] , Et @ X2[k] ) ) ** 2 )

                    if (d_x2_x1 + d_x1_x2) < eps:
                        inliers.append(k)

                inliers = np.array(inliers)
                inliers = np.concatenate((sample_indices, inliers))

                if inliers.shape[0] > best_num_inliers:
                    best_num_inliers = inliers.shape[0]

                    best_E = E
                    best_inliers = inliers

        except:
            traceback.print_exc()

        return best_E, best_inliers