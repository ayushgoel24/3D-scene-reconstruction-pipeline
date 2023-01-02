from environment import ApplicationProperties
from src.sfm2view import ReconstructionFrom2DImages
from src.flow import ComputeFlow

applicationProperties = ApplicationProperties("application.yml")
applicationProperties.initializeProperties()

## 3D Reconstruction from 2D Images ##
reconstructionFrom2DImages = ReconstructionFrom2DImages( applicationProperties )
reconstructionFrom2DImages.complete_pipeline()

opticalFlow = ComputeFlow( applicationProperties )
opticalFlow.complete_pipeline()