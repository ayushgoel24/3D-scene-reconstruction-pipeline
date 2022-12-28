import numpy as np

class PoseEstimation( object ):

    def __init__(self) -> None:
        pass

    @staticmethod
    def pose_candidates_from_E(E):
        transform_candidates = []
        ##Note: each candidate in the above list should be a dictionary with keys "T", "R"
        
        U, S, Vt = np.linalg.svd(E)
        R_90 = np.array( [ [ 0, -1, 0 ], [ 1, 0, 0 ], [ 0, 0, 1 ] ] )
        R_90_n = np.array( [ [ 0, 1, 0 ], [ -1, 0, 0 ], [ 0, 0, 1 ] ] )

        T_1 = U[:, 2] / np.linalg.norm( U[:, 2] )
        T_2 = -1 * T_1

        R_1 = U @ R_90.T @ Vt
        R_2 = U @ R_90_n.T @ Vt

        transform_candidates.append({"T": T_1, "R": R_1})
        transform_candidates.append({"T": T_1, "R": R_2})
        transform_candidates.append({"T": T_2, "R": R_1})
        transform_candidates.append({"T": T_2, "R": R_2})

        return transform_candidates