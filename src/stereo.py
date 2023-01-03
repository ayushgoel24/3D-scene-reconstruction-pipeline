from environment import ApplicationProperties
from src.multiviewstereo import DataLoader
import matplotlib.pyplot as plt

class Stereo( object ):

    def __init__( self, applicationProperties: ApplicationProperties ) -> None:
        self.applicationProperties = applicationProperties
        self.dataDirectory = applicationProperties.get_property_value( "Stereo.dataDirectory" )

    def load_dataset( self ):
        # reference: https://vision.middlebury.edu/mview/
        DATA = DataLoader.load_middlebury_data( self.dataDirectory )
        view_i, view_j = DATA[0], DATA[3]
        plt.subplot(1, 2, 1)
        plt.title("Sample: left view (reference)")
        plt.imshow(view_i["rgb"])
        plt.subplot(1, 2, 2)
        plt.title("Sample: right view")
        plt.imshow(view_j["rgb"])
        plt.show()

    def complete_pipeline( self ) -> None:

        EPS = 1e-8

