import matplotlib.pyplot as plt
import numpy as np

class Reprojections( object ):

    def __init__(self) -> None:
        pass

    def show_reprojections(image1, image2, uncalibrated_1, uncalibrated_2, P1, P2, K, T, R, plot=True):

        P1proj = np.zeros(( P1.shape[0], 3 ))
        P2proj = np.zeros(( P1.shape[0], 3 ))

        for i in range( P1.shape[0] ):
            P1proj[i, :] = K @ ( R @ P1.T[:, i] + T)
            P2proj[i, :] = K @ np.linalg.inv(R) @ ( P2.T[:, i] - T )

        if (plot):
            plt.figure(figsize=(6.4*3, 4.8*3))
            ax = plt.subplot(1, 2, 1)
            ax.set_xlim([0, image1.shape[1]])
            ax.set_ylim([image1.shape[0], 0])
            plt.imshow(image1[:, :, ::-1])
            plt.plot(P2proj[:, 0] / P2proj[:, 2],
                P2proj[:, 1] / P2proj[:, 2], 'bs')
            plt.plot(uncalibrated_1[0, :], uncalibrated_1[1, :], 'ro')

            ax = plt.subplot(1, 2, 2)
            ax.set_xlim([0, image1.shape[1]])
            ax.set_ylim([image1.shape[0], 0])
            plt.imshow(image2[:, :, ::-1])
            plt.plot(P1proj[:, 0] / P1proj[:, 2],
                P1proj[:, 1] / P1proj[:, 2], 'bs')
            plt.plot(uncalibrated_2[0, :], uncalibrated_2[1, :], 'ro')
            
        else:
            return P1proj, P2proj