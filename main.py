from environment import ApplicationProperties
from src.sfm2view import ReconstructionFrom2DImages

applicationProperties = ApplicationProperties("application.yml")
applicationProperties.initializeProperties()

## 3D Reconstruction from 2D Images ##
reconstructionFrom2DImages = ReconstructionFrom2DImages( applicationProperties )
reconstructionFrom2DImages.complete_pipeline()