import numpy as np

class FlowComputation( object ):

    @staticmethod
    def getWinBound(img_sz, startX, startY, win_size):
        szY, szX = img_sz
        
        win_left = startX - (win_size - 1) // 2
        win_right = startX + (win_size + 1) // 2 - 1

        if win_left < 0: win_left = 0
        elif win_right > szX: win_right = szX
            
        win_top = startY - (win_size - 1) // 2
        win_bottom = startY + (win_size + 1) // 2 - 1

        if win_top < 0: win_top = 0
        elif win_bottom > szY: win_bottom = szY

        return int(win_left), int(win_right), int(win_top), int(win_bottom)

    @staticmethod
    def flow_lk_patch(Ix, Iy, It, x, y, size=5):
        """
        params:
            @Ix: np.array(h, w)
            @Iy: np.array(h, w)
            @It: np.array(h, w)
            @x: int
            @y: int
        return value:
            flow: np.array(2,)
            conf: np.array(1,)
        """

        win_left, win_right, win_top, win_bottom = FlowComputation.getWinBound( It.shape, x, y, size )

        windowed_Ix = Ix[ win_top : win_bottom + 1, win_left : win_right + 1 ]
        windowed_Iy = Iy[ win_top : win_bottom + 1, win_left : win_right + 1 ]
        windowed_It = It[ win_top : win_bottom + 1, win_left : win_right + 1 ]

        A=[]
        b=[]
        for i in range( windowed_Ix.shape[1] ):
            for j in range( windowed_Ix.shape[0] ):
                A.append( [ windowed_Ix[j, i], windowed_Iy[j, i] ] )
                b.append( -windowed_It[j, i] )

        A = np.array( A )
        b = np.array( b )
        
        # Ax = b
        flow, _, _, s = np.linalg.lstsq( A, b, rcond=None )
        flow = flow.reshape((2, ))

        conf = np.min( s )
        conf = conf.reshape((1,))

        return flow, conf


    @staticmethod
    def flow_lk(Ix, Iy, It, size=5):
        """
        params:
            @Ix: np.array(h, w)
            @Iy: np.array(h, w)
            @It: np.array(h, w)
        return value:
            flow: np.array(h, w, 2)
            conf: np.array(h, w)
        """

        image_flow = np.zeros([ Ix.shape[0], Ix.shape[1], 2 ])
        confidence = np.zeros([ Ix.shape[0], Ix.shape[1] ])
        for x in range( Ix.shape[1] ):
            for y in range( Ix.shape[0] ):
                flow, conf = FlowComputation.flow_lk_patch( Ix, Iy, It, x, y )
                image_flow[y, x, :] = flow
                confidence[y, x] = conf
        return image_flow, confidence