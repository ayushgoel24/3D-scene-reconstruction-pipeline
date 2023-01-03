import numpy as np
import os.path as osp
import os
from tqdm import tqdm
import imageio

class DataLoader( object ):
    
    @staticmethod
    def load_middlebury_data( datadir ):
        """
        "imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3"
            The projection matrix for that image is given by K*[R t]
        """

        # from the dataset readme
        BBox = {
            "templeRing": np.array([[-0.023121, -0.038009, -0.091940], [0.078626, 0.121636, -0.017395]])
        }

        camera_fn = [osp.join(datadir, fn) for fn in os.listdir(datadir) if fn.endswith("_par.txt")]
        assert len(camera_fn) == 1, "camera not found or duplicated"
        
        viz_fn = [osp.join(datadir, fn) for fn in os.listdir(datadir) if fn.endswith("_ang.txt")]
        assert len(viz_fn) == 1, "camera not found or duplicated"
        
        with open(camera_fn[0]) as f:
            cam_data = f.readlines()

        with open(viz_fn[0]) as f:
            ang_data = f.readlines()

        n_views = int(cam_data.pop(0))

        DATA = []
        for cam, ang in tqdm( zip( cam_data, ang_data ) ):
            l = cam[:-1].split(" ")
            image_fn = l.pop(0)
            l = np.array( l )
            _K, _R, _t = l[:9].reshape(3, 3), l[9:18].reshape(3, 3), l[18:]
            lat, lon = ang.split(" ")[:-1]
            lat, lon = float(lat), float(lon)
            image = imageio.imread( osp.join( datadir, image_fn ) )
            DATA.append(
                {
                    "K": _K.astype(np.float),
                    "R": _R.astype(np.float),
                    "T": _t.astype(np.float),
                    "lat": lat,
                    "lon": lon,
                    "rgb": image,
                }
            )
        assert len(DATA) == n_views
        return DATA