import numpy as np

# Class to define 2x2 rubiks cube
class cube:
    '''
    Initialize cube state (starts in solved state)
    Cube state is defined by 6 faces, each a 2 x 2 matrix 

    '''

    def __init__(self,size=2):

        self.size = size
        self.cubeState = np.array([size**2*['b'],size**2*['w'],size**2*['y'],
                                size**2*['r'],size**2*['o'],size**2*['g']]
                                ).reshape(6,size,size)
        
        self.front = np.array(['b','b','b','b']).reshape(2,2)
        self.top = np.array(['w','w','w','w']).reshape(2,2)
        self.bottom = np.array(['y','y','y','y']).reshape(2,2)
        self.left = np.array(['r','r','r','r']).reshape(2,2)
        self.right = np.array(['o','o','o','o']).reshape(2,2)
        self.back = np.array(['g','g','g','g']).reshape(2,2)


    # Randomize faces
    def randomize(self):


    # Rotate front face clockwise. 
    def clockwise(self, cubeState):
        # Only 2x2 so only need to operate on front face. This is
        # equivalent to doing the opposite rotation on the back face.

        '''
        
        '''

        # Copy array
        new_array = copy.deepcopy(array)

        # Rotate outer
        # Rotate 1,0 to 0,1
        new_array[0][1][1][1] = array[1][0][0][1]
        new_array[0][1][1][0] = array[1][0][1][1]
        # Rotate 0,1 to 1,2
        new_array[1][2][0][0] = array[0][1][1][0]
        new_array[1][2][1][0] = array[0][1][1][1]
        # Rotate 1,2 to 2,1
        new_array[2][1][0][1] = array[1][2][0][0]
        new_array[2][1][0][0] = array[1][2][1][0]
        # Rotate 1,2 to 2,1
        new_array[1][0][1][1] = array[2][1][0][1]
        new_array[1][0][0][1] = array[2][1][0][0]

        # Rotate main face
        new_array = np.rot90()

        return new_array


    # Rotate front face counterclockwise
    def counterclockwise(self):



    # Rotate lefthand square forward
    def forward(self):
        # Only 2x2 so only need to operate on left face. This is
        # equivalent to doing the opposite rotation on the right face.


    # Rotate lefthand square backward
    def backward(self):