import argparse
from environment import ApplicationProperties
from src.sfm2view import ReconstructionFrom2DImages
from src.flow import ComputeFlow
from src.stereo import Stereo

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", action="store_true")

    args = parser.parse_args()

    applicationProperties = ApplicationProperties("application.yml")
    applicationProperties.initializeProperties()

    if args.pipeline == "SfM":
        ## 3D Reconstruction from 2D Images ##
        reconstructionFrom2DImages = ReconstructionFrom2DImages( applicationProperties )
        reconstructionFrom2DImages.complete_pipeline()

    elif args.pipeline == "flow":
        opticalFlow = ComputeFlow( applicationProperties )
        opticalFlow.complete_pipeline()

    else:
        stereo = Stereo( applicationProperties )
        stereo.complete_pipeline()