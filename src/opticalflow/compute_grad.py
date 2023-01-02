import numpy as np
from scipy.ndimage import convolve1d

class GradientComputation( object ):

    KERNEL_G = np.array([0.015625, 0.093750, 0.234375, 0.312500, 0.234375, 0.093750, 0.015625])
    KERNEL_H = np.array([0.03125, 0.12500, 0.15625, 0, -0.15625, -0.1250, -0.03125])

    @staticmethod
    def compute_Ix( imgs ):
        Ix = convolve1d( imgs, GradientComputation.KERNEL_H, axis=1 )
        Ix = convolve1d( Ix, GradientComputation.KERNEL_G, axis=0 )
        Ix = convolve1d( Ix, GradientComputation.KERNEL_G, axis=2 )
        return Ix

    @staticmethod
    def compute_Iy( imgs ):
        Iy = convolve1d( imgs, GradientComputation.KERNEL_H, axis=0 )
        Iy = convolve1d( Iy, GradientComputation.KERNEL_G, axis=1 )
        Iy = convolve1d( Iy, GradientComputation.KERNEL_G, axis=2 )
        return Iy

    @staticmethod
    def compute_It( imgs ):
        It = convolve1d( imgs, GradientComputation.KERNEL_H, axis=2 )
        It = convolve1d( It, GradientComputation.KERNEL_G, axis=0 )
        It = convolve1d( It, GradientComputation.KERNEL_G, axis=1 )
        return It