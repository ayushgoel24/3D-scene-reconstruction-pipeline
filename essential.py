import numpy as np

class EssentialMatrix( object ):

    def __init__(self) -> None:
        pass

    @staticmethod
    def compute_using_least_squares_estimation( X1, X2 ):
        
        A = np.hstack(( ( X1[:, 0] * X2[:, :].T ).T, ( X1[:, 1] * X2[:, :].T ).T, ( X1[:, 2] * X2[:, :].T ).T ))
        [ _, _, Vt ] = np.linalg.svd( A )
        E = Vt[-1, :].reshape( 3, 3 ).T
        [ U, _, Vt ] = np.linalg.svd( E )
        E = U @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) @ Vt

        return E