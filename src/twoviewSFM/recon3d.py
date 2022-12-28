import numpy as np
import traceback

class ThreeDReconstruction( object ):

    def __init__(self) -> None:
         pass

    @staticmethod
    def reconstruct3D(transform_candidates, calibrated_1, calibrated_2):
        """This functions selects (T,R) among the 4 candidates transform_candidates
        such that all triangulated points are in front of both cameras.
        """
        try:
            best_num_front = -1
            best_candidate = None
            best_lambdas = None
            for candidate in transform_candidates:
                R = candidate['R']
                T = candidate['T']

                lambdas = np.zeros((2, calibrated_1.shape[0]))

                for k in range(calibrated_1.shape[0]):
                    A = np.hstack((calibrated_2[k].T.reshape(-1, 1), (-R @ calibrated_1[k].T).reshape(-1, 1)))
                    lambdas[:,k] = np.linalg.inv( A.T @ A ) @ A.T @ T

                num_front = np.sum(np.logical_and(lambdas[0]>0, lambdas[1]>0))

                if num_front > best_num_front:
                    best_num_front = num_front
                    best_candidate = candidate
                    best_lambdas = lambdas
                    print("best", num_front, best_lambdas[0].shape)
                else:
                    print("not best", num_front)


            P1 = best_lambdas[1].reshape(-1, 1) * calibrated_1
            P2 = best_lambdas[0].reshape(-1, 1) * calibrated_2
            T = best_candidate['T']
            R = best_candidate['R']

        except:
            traceback.print_exc()

        return P1, P2, T, R