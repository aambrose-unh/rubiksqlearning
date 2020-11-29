import copy
import itertools
import numpy as np

# Class to define 2x2 rubiks cube
class cube:
    '''
    Initialize cube state (starts in solved state)
    Cube state is defined by 6 faces, each a 2 x 2 matrix 

    '''

    def __init__(self,size=2):

        self.size = size

        self.front = np.array(size**2*['b']).reshape(size,size)
        self.top = np.array(size**2*['w']).reshape(size,size)
        self.bottom = np.array(size**2*['y']).reshape(size,size)
        self.left = np.array(size**2*['r']).reshape(size,size)
        self.right = np.array(size**2*['o']).reshape(size,size)
        self.back = np.array(size**2*['g']).reshape(size,size)

        self.cubeState = np.array([self.front,self.top,self.bottom,
                                self.left,self.right,self.back]
                                ).reshape(6,size,size)

        self.actionList = [self.clockwise, self.counterclockwise, self.forward,
                        self.backward, self.toLeft, self.toRight]

        self.al = [cube.clockwise,cube.counterclockwise,cube.forward,
                    cube.backward,cube.toLeft,cube.toRight]

        self.faces = {'front':self.front,'top':self.top,'bottom':self.bottom,
                                'left':self.left,'right':self.right,'back':self.back}
    
    def update_cubeState(self):
        self.cubeState = np.array([self.front,self.top,self.bottom,
                                self.left,self.right,self.back]
                                ).reshape(6,self.size,self.size)
        self.faces = {'front':self.front,'top':self.top,'bottom':self.bottom,
                                'left':self.left,'right':self.right,'back':self.back}

    #===============================================
    # Get Features
    #===============================================

    def get_features(self):
        feat_vector = dict()
        tot_cpf = 0
        one_color_face = 0
        for face in self.faces:
            # calc colors per face
            num_col = len(np.unique(self.faces[face]))
            tot_cpf += num_col
            # Add 1/color for that face to feature vector
            feat_vector[f'inv_cpf_{face}'] = (1/num_col)
            # get total single colored faces
            if num_col == 1:
                one_color_face += 1
        # Add 1 / average colors per face
        feat_vector['inv_avg_cpf'] = (1/(tot_cpf/6))
        # Add number of single colored faces (divided by 6 to keep same scale as other features)
        feat_vector['one_color_faces'] = (one_color_face/6)

        # Interaction Features
        for i in itertools.combinations(self.faces,r=2):
            num_col0 = 1/len(np.unique(self.faces[i[0]]))
            num_col1 = 1/len(np.unique(self.faces[i[1]]))
            feat_vector[i[0]+'_'+i[1]] = num_col0*num_col1

        # Corners
        feat_vector['flt'] = len(np.unique([self.front[0,0],self.left[0,1],self.top[1,0]]))/3
        feat_vector['frt'] = len(np.unique([self.front[0,1],self.right[0,0],self.top[1,1]]))/3
        feat_vector['flb'] = len(np.unique([self.front[1,0],self.left[1,1],self.bottom[0,0]]))/3
        feat_vector['frb'] = len(np.unique([self.front[1,1],self.right[1,0],self.bottom[0,1]]))/3
        feat_vector['blt'] = len(np.unique([self.back[0,1],self.left[0,0],self.top[0,0]]))/3
        feat_vector['brt'] = len(np.unique([self.back[0,0],self.right[0,1],self.top[0,1]]))/3
        feat_vector['blb'] = len(np.unique([self.back[1,1],self.left[1,0],self.bottom[1,0]]))/3
        feat_vector['brb'] = len(np.unique([self.back[1,0],self.right[1,1],self.bottom[1,1]]))/3

        # Maybe check for patterns of previous moves????

        return feat_vector

    def copy(self):
        return copy.deepcopy(self)

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
    def randomize(self,num_actions=15):
        '''
        Performs num_actions number of actions randomly selected from
        self.actionList
        '''
        
        for num in range(num_actions):
            a = np.random.random_integers(0,len(self.actionList)-1)
            # print('A = ',a)
            # print('Action - ',self.actionList[a])
            self.actionList[a]()

        self.update_cubeState()
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
        self.left[:,1] = refState.bottom[0,:]
        self.top[1,:] = refState.left[:,1]
        self.right[:,0] = refState.top[1,:]
        self.bottom[0,:] = refState.right[:,0]

        # Rotate main face
        self.front = np.rot90(refState.front,axes=(1,0))
        self.update_cubeState()
        # return "To be implemented"


    # Rotate front face counterclockwise
    def counterclockwise(self):
        '''
        Modifies cubeState in place to rotate front face 90deg counter-clockwise
        '''

        # Copy array to keep as reference
        refState = self.copy()
        # Rotate outer
        self.left[:,1] = refState.top[1,:]
        self.top[1,:] = refState.right[:,0]
        self.right[:,0] = refState.bottom[0,:]
        self.bottom[0,:] = refState.left[:,1]

        # Rotate main face
        self.front = np.rot90(refState.front, axes=(0,1))
        self.update_cubeState()
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
        self.front[:,0] = refState.top[:,0]
        self.top[:,0] = refState.back[:,1]
        self.back[:,1] = refState.bottom[:,0]
        self.bottom[:,0] = refState.front[:,0]

        # Rotate main face
        self.left = np.rot90(refState.left, axes=(1,0))
        self.update_cubeState()
        # return "To be implemented"

    # Rotate lefthand square backward
    def backward(self):

        # Copy array to keep as reference
        refState = self.copy()
        # Rotate outer
        self.front[:,0] = refState.bottom[:,0]
        self.top[:,0] = refState.front[:,0]
        self.back[:,1] = refState.top[:,0]
        self.bottom[:,0] = refState.back[:,1]

        # Rotate main face
        self.left = np.rot90(refState.left, axes=(0,1))
        self.update_cubeState()

        # return "To be implemented"

    # Rotate top square to the right
    def toRight(self):
        # Only 2x2 so only need to operate on left face. This is
        # equivalent to doing the opposite rotation on the right face.

        '''
        Modifies cubeState in place to rotate left face forward (top-left corner
        rotating towards user and downward)
        '''

        # Copy array to keep as reference
        refState = self.copy()
        # Rotate outer
        self.front[0,:] = refState.left[0,:]
        self.left[0,:] = refState.back[0,:]
        self.back[0,:] = refState.right[0,:]
        self.right[0,:] = refState.front[0,:]

        # Rotate main face
        self.top = np.rot90(refState.top, axes=(1,0))
        self.update_cubeState()
        # return "To be implemented"

    # Rotate top square to the left
    def toLeft(self):

        # Copy array to keep as reference
        refState = self.copy()
        # Rotate outer
        self.front[0,:] = refState.right[0,:]
        self.right[0,:] = refState.back[0,:]
        self.back[0,:] = refState.left[0,:]
        self.left[0,:] = refState.front[0,:]

        # Rotate main face
        self.top = np.rot90(refState.top, axes=(0,1))
        self.update_cubeState()
        # return "To be implemented"