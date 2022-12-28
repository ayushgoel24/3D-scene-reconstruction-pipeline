import matplotlib.pyplot as plt
import numpy as np 

class Epipoles( object ):

    def __init__(self) -> None:
        pass

    @staticmethod
    def plot_lines( lines, h, w ):
        """ Utility function to plot lines
        """

        for i in range(lines.shape[1]):
            # plt.close('all')
            if abs(lines[0, i] / lines[1, i]) < 1:
                y0 = -lines[2, i] / lines[1, i]
                yw = y0 - w * lines[0, i] / lines[1, i]
                plt.plot([0, w], [y0, yw])
                # plt.clf()
                # plt.show()
                # plt.savefig(filename)
            else:
                x0 = -lines[2, i] / lines[0, i]
                xh = x0 - h * lines[1, i] / lines[0, i]
                plt.plot([x0, xh], [0, h])
                # plt.clf
                # plt.show()
                # plt.savefig(filename)

    @staticmethod
    def plot_epipolar_lines(image1, image2, uncalibrated_1, uncalibrated_2, E, K, plot=True):
        """ Plots the epipolar lines on the images
        """

        F = np.linalg.inv( K ).T @ E @ np.linalg.inv( K )
        epipolar_lines_in_1 = F.T @ uncalibrated_2
        epipolar_lines_in_2 = F @ uncalibrated_1
        
        if(plot):

            plt.figure(figsize=(6.4*3, 4.8*3))
            ax = plt.subplot(1, 2, 1)
            ax.set_xlim([0, image1.shape[1]])
            ax.set_ylim([image1.shape[0], 0])
            plt.imshow(image1[:, :, ::-1])
            Epipoles.plot_lines(epipolar_lines_in_1, image1.shape[0], image1.shape[1])

            ax = plt.subplot(1, 2, 2)
            ax.set_xlim([0, image1.shape[1]])
            ax.set_ylim([image1.shape[0], 0])
            plt.imshow(image2[:, :, ::-1])
            Epipoles.plot_lines(epipolar_lines_in_2, image2.shape[0], image2.shape[1])
            
        else:
            return epipolar_lines_in_1, epipolar_lines_in_2