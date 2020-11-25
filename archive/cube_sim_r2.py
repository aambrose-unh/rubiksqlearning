import copy
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
    
        self.front = self.cubeState[0]
        self.top = self.cubeState[1]
        self.bottom = self.cubeState[2]
        self.left = self.cubeState[3]
        self.right = self.cubeState[4]
        self.back = self.cubeState[5]

        self.actionList = [self.clockwise, self.counterclockwise,
                            self.forward, self.backward]

    def copy(self):
        cubeCopy = cube()
        cubeCopy.front = self.front
        cubeCopy.top = self.top
        cubeCopy.bottom = self.bottom
        cubeCopy.left = self.left
        cubeCopy.right = self.right
        cubeCopy.back = self.back

    def getFace(self,cubeState,face):
        if face == 'front':
            return cubeState[0]
        if face == 'top':
            return cubeState[1]
        if face == 'bottom':
            return cubeState[2]
        if face == 'left':
            return cubeState[3]
        if face == 'right':
            return cubeState[4]
        if face == 'back':
            return cubeState[5]

    # Show flattened cube
    def showCube(self):
        forShow = np.full((3,4,2,2),' ')
        forShow[0,1] = self.top
        forShow[1,0] = self.left
        forShow[1,1] = self.front
        forShow[1,2] = self.right
        forShow[1,3] = self.back
        forShow[2,1] = self.bottom
        def printRow(row):
            for i in row:
                print(i[0,:],end=' ')
            print('')
            for i in row:
                print(i[1,:],end=' ')
        for row in range(len(forShow)):
            print("\n")
            printRow(forShow[row])

    
    # Randomize faces
    def randomize(self,num_actions=5):
        '''
        Performs num_actions number of actions randomly selected from
        self.actionList
        '''
        
        for num in range(num_actions):
            a = np.random.random_integers(0,len(self.actionList)-1)
            print('A = ',a)
            print('Action - ',self.actionList[a])
            self.actionList[a]()

        # return "To be implemented"

    # Rotate front face clockwise. 
    def clockwise(self):
        # Only 2x2 so only need to operate on front face. This is
        # equivalent to doing the opposite rotation on the back face.

        '''
        Modifies cubeState in place to rotate front face 90deg clockwise
        '''

        # Copy array to keep as reference
        refState = self.copy()
        # Rotate outer
        self.left[:,1] = self.getFace(refState,'bottom')[0,:]
        self.top[1,:] = self.getFace(refState,'left')[:,1]
        self.right[:,0] = self.getFace(refState,'top')[1,:]
        self.bottom[0,:] = self.getFace(refState,'right')[:,0]

        # Rotate main face
        self.front = np.rot90(self.getFace(refState,'front'),axes=(1,0))

        # return "To be implemented"


    # Rotate front face counterclockwise
    def counterclockwise(self):
        '''
        Modifies cubeState in place to rotate front face 90deg counter-clockwise
        '''

        # Copy array to keep as reference
        refState = self.copy()
        # Rotate outer
        self.left[:,1] = self.getFace(refState,'top')[1,:]
        self.top[1,:] = self.getFace(refState,'right')[:,0]
        self.right[:,0] = self.getFace(refState,'bottom')[0,:]
        self.bottom[0,:] = self.getFace(refState,'left')[:,1]

        # Rotate main face
        self.front = np.rot90(self.getFace(refState,'front'), axes=(0,1))

        # return "To be implemented"


    # Rotate lefthand square forward
    def forward(self):
        # Only 2x2 so only need to operate on left face. This is
        # equivalent to doing the opposite rotation on the right face.

        '''
        Modifies cubeState in place to rotate left face forward (top-left corner
        rotating towards user and downward)
        '''

        # Copy array to keep as reference
        refState = self.copy()
        # Rotate outer
        self.front[:,0] = self.getFace(refState,'top')[:,0]
        self.top[:,0] = self.getFace(refState,'back')[:,1]
        self.back[:,1] = self.getFace(refState,'bottom')[:,0]
        self.bottom[:,0] = self.getFace(refState,'front')[:,0]

        # Rotate main face
        self.left = np.rot90(self.getFace(refState,'left'), axes=(1,0))

        # return "To be implemented"

    # Rotate lefthand square backward
    def backward(self):

        # Copy array to keep as reference
        refState = self.copy()
        # Rotate outer
        self.front[:,0] = self.getFace(refState,'bottom')[:,0]
        self.top[:,0] = self.getFace(refState,'front')[:,0]
        self.back[:,1] = self.getFace(refState,'top')[:,0]
        self.bottom[:,0] = self.getFace(refState,'back')[:,1]

        # Rotate main face
        self.left = np.rot90(self.getFace(refState,'left'), axes=(0,1))


        return "To be implemented"