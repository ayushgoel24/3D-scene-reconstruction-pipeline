from environment import ApplicationProperties
import cv2
import os
import numpy as np
from opticalflow import GradientComputation, FlowComputation, VisualiseFlow, EpipoleComputation, DepthComputation, PlanarParameters

class ComputeFlow( object ):
    
    def __init__( self, applicationProperties: ApplicationProperties ) -> None:
        self.applicationProperties = applicationProperties
        self.dataDirectory = applicationProperties.get_property_value( "OpticalFlow.dataDirectory" )

    def load_images( self ) -> np.ndarray:
        images = [ cv2.imread( os.path.join( self.dataDirectory, "insight{}.png".format( i ) ), cv2.IMREAD_GRAYSCALE ).astype(float) for i in range( 20, 27 ) ]
        images = np.stack( images, axis=-1 )
        return images

    def compute_gradients( self, images ) -> tuple:
        Ix = GradientComputation.compute_Ix( images )
        Iy = GradientComputation.compute_Iy( images )
        It = GradientComputation.compute_It( images )
        return ( Ix, Iy, It )

    def compute_flow( self, Ix, Iy, It, valid_idx ) -> tuple:
        flow, confidence = FlowComputation.flow_lk( Ix[..., valid_idx], Iy[..., valid_idx], It[..., valid_idx] )
        return ( flow, confidence )

    def find_epipoles( self, flow, confidence ) -> tuple:
        block_mask = np.array( confidence )
        ep, inliers = EpipoleComputation.epipole( flow[:, :, 0], flow[:, :, 1], block_mask, self.applicationProperties.get_property_value( "OpticalFlow.threshold" ), num_iterations = 1000 )
        ep = ep / ep[2]
        return ( block_mask, ep )

    def compute_depth_map( self, flow, confidence, ep, K ) -> np.array:
        depth_map = DepthComputation.depth(flow, confidence, ep, K, thres=self.applicationProperties.get_property_value( "OpticalFlow.threshold" ))
        return depth_map

    def complete_pipeline( self ) -> None:
        images = self.load_images()

        Ix, Iy, It = self.compute_gradients( images )

        # only take the image in the middle
        valid_idx = 3
        flow, confidence = self.compute_flow( Ix, Iy, It, valid_idx )

        K = np.array([[1118, 0, 357],
                    [0, 1121, 268],
                    [0, 0, 1]])

        if bool( self.applicationProperties.get_property_value( "OpticalFlow.plotFlow" ) ):
            # plt.figure()
            VisualiseFlow.plot_flow( images[..., valid_idx], flow, confidence, threshmin=int( self.applicationProperties.get_property_value( "OpticalFlow.threshold" ) ) )
            # plt.savefig(f"flow_{args.threshmin}.png")
            # plt.show()

        if bool( self.applicationProperties.get_property_value( "OpticalFlow.findEpipoles" ) or self.applicationProperties.get_property_value( "OpticalFlow.computeDepth" ) ):
            block_mask, ep = self.find_epipoles( flow, confidence )
            # plt.figure()
            # plot_flow(images[..., valid_idx], flow, block_mask, threshmin=args.threshmin)
            # plt.scatter(ep[0]+512//2, ep[1]+512//2, c='r')
            # x = np.array([i for i in range(512)])
            # xv, yv = np.meshgrid(x, x)
            # xp = np.stack([xv.flatten(),yv.flatten(),np.ones((512,512)).flatten()]).T
            # plt.scatter(xp[inliers,:][:,0], xp[inliers,:][:,1], c='b', s=0.1)
            # plt.savefig(f"epipole_{args.threshmin}.png")
            # plt.show()

        if bool( self.applicationProperties.get_property_value( "OpticalFlow.computeDepth" ) ):
            depth_map = self.compute_depth_map(flow, confidence, ep, K)
            # sns.heatmap(depth_map, square=True, cmap='mako')
            # plt.savefig(f"depth_{args.threshmin}.png")
            # plt.show()

        if bool( self.applicationProperties.get_property_value( "OpticalFlow.computePlanarFlow" ) ):
            up = [312, 0]
            down = [512, 200]
            params = PlanarParameters.compute_planar_params( flow[..., 0], flow[..., 1], K, up=up, down=down )
            print("8 Arguments are: ", params)