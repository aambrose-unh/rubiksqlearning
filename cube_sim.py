import copy
import numpy as np

# Class to define 2x2 rubiks cube


class cube:
    """
    Initialize cube state (starts in solved state)
    Cube state is defined by 6 faces, each a 2 x 2 matrix

    """

    def __init__(self, size=2):

        self.size = size

        self.front = np.array(size ** 2 * ["b"]).reshape(size, size)
        self.top = np.array(size ** 2 * ["w"]).reshape(size, size)
        self.bottom = np.array(size ** 2 * ["y"]).reshape(size, size)
        self.left = np.array(size ** 2 * ["r"]).reshape(size, size)
        self.right = np.array(size ** 2 * ["o"]).reshape(size, size)
        self.back = np.array(size ** 2 * ["g"]).reshape(size, size)

        self.cubeState = np.array(
            [self.front, self.top, self.bottom, self.left, self.right, self.back]
        ).reshape(6, size, size)

        self.actionList = [
            self.clockwise,
            self.counterclockwise,
            self.forward,
            self.backward,
            self.toLeft,
            self.toRight,
        ]

        self.al = [
            cube.clockwise,
            cube.counterclockwise,
            cube.forward,
            cube.backward,
            cube.toLeft,
            cube.toRight,
        ]

        self.faces = {
            "front": self.front,
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right,
            "back": self.back,
        }

        self.adjacent = {
            "front": [
                self.top[1, :],
                self.right[:, 0],
                self.bottom[0, :],
                self.left[:, 1],
            ],
            "top": [
                self.front[0, :],
                self.right[0, :],
                self.back[0, :],
                self.left[0, :],
            ],
            "bottom": [
                self.front[1, :],
                self.right[1, :],
                self.back[1, :],
                self.left[1, :],
            ],
            "left": [
                self.front[:, 0],
                self.top[:, 0],
                self.bottom[:, 0],
                self.back[:, 1],
            ],
            "right": [
                self.front[:, 1],
                self.top[:, 1],
                self.bottom[:, 1],
                self.back[:, 0],
            ],
            "back": [
                self.top[0, :],
                self.right[:, 1],
                self.bottom[1, :],
                self.left[:, 0],
            ],
        }

    def update_cubeState(self):
        """
        Helper function to update cubeState and faces dictionary based on face values
        Individual faces are modified in place, but cubeState and faces wont update unless forced
        """
        self.cubeState = np.array(
            [self.front, self.top, self.bottom, self.left, self.right, self.back]
        ).reshape(6, self.size, self.size)
        self.faces = {
            "front": self.front,
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right,
            "back": self.back,
        }
        self.adjacent = {
            "front": [
                self.top[1, :],
                self.right[:, 0],
                self.bottom[0, :],
                self.left[:, 1],
            ],
            "top": [
                self.front[0, :],
                self.right[0, :],
                self.back[0, :],
                self.left[0, :],
            ],
            "bottom": [
                self.front[1, :],
                self.right[1, :],
                self.back[1, :],
                self.left[1, :],
            ],
            "left": [
                self.front[:, 0],
                self.top[:, 0],
                self.bottom[:, 0],
                self.back[:, 1],
            ],
            "right": [
                self.front[:, 1],
                self.top[:, 1],
                self.bottom[:, 1],
                self.back[:, 0],
            ],
            "back": [
                self.top[0, :],
                self.right[:, 1],
                self.bottom[1, :],
                self.left[:, 0],
            ],
        }

    def copy(self):
        """Helper function to make deep copy of cube"""
        return copy.deepcopy(self)

    # Show flattened cube
    def showCube(self):
        """
        Prints cube in flattened state
        """
        forShow = np.full((3, 4, 2, 2), " ")
        forShow[0, 1] = self.top
        forShow[1, 0] = self.left
        forShow[1, 1] = self.front
        forShow[1, 2] = self.right
        forShow[1, 3] = self.back
        forShow[2, 1] = self.bottom

        def printRow(row):
            for i in row:
                print(i[0, :], end=" ")
            print("")
            for i in row:
                print(i[1, :], end=" ")

        for row in range(len(forShow)):
            print("\n")
            printRow(forShow[row])

    # Randomize faces
    def randomize(self, num_actions=15):
        """
        Performs num_actions number of actions randomly selected from
        self.actionList
        """

        for num in range(num_actions):
            a = np.random.random_integers(0, len(self.actionList) - 1)
            self.actionList[a]()

        self.update_cubeState()

    # Rotate front face clockwise.
    def clockwise(self):
        # Only 2x2 so only need to operate on front face. This is
        # equivalent to doing the opposite rotation on the back face.

        """
        Modifies cubeState in place to rotate front face 90deg clockwise
        """

        # Copy array to keep as reference
        refState = self.copy()
        # Rotate outer
        self.left[:, 1] = refState.bottom[0, :]
        self.top[1, :] = np.flip(refState.left[:, 1])
        self.right[:, 0] = refState.top[1, :]
        self.bottom[0, :] = np.flip(refState.right[:, 0])

        # Rotate main face
        self.front = np.rot90(refState.front, axes=(1, 0))
        self.update_cubeState()

    # Rotate front face counterclockwise
    def counterclockwise(self):
        """
        Modifies cubeState in place to rotate front face 90deg counter-clockwise
        """

        # Copy array to keep as reference
        refState = self.copy()
        # Rotate outer
        self.left[:, 1] = np.flip(refState.top[1, :])
        self.top[1, :] = refState.right[:, 0]
        self.right[:, 0] = np.flip(refState.bottom[0, :])
        self.bottom[0, :] = refState.left[:, 1]

        # Rotate main face
        self.front = np.rot90(refState.front, axes=(0, 1))
        self.update_cubeState()

    # Rotate lefthand square forward
    def forward(self):
        # Only 2x2 so only need to operate on left face. This is
        # equivalent to doing the opposite rotation on the right face.

        """
        Modifies cubeState in place to rotate left face forward (top-left corner
        rotating towards user and downward)
        """

        # Copy array to keep as reference
        refState = self.copy()
        # Rotate outer
        self.front[:, 0] = refState.top[:, 0]
        self.top[:, 0] = np.flip(refState.back[:, 1])
        self.back[:, 1] = np.flip(refState.bottom[:, 0])
        self.bottom[:, 0] = refState.front[:, 0]

        # Rotate main face
        self.left = np.rot90(refState.left, axes=(1, 0))
        self.update_cubeState()

    # Rotate lefthand square backward
    def backward(self):
        """
        Modifies cubeState in place to rotate left face backward (top-left corner
        rotating away from user)
        """

        # Copy array to keep as reference
        refState = self.copy()
        # Rotate outer
        self.front[:, 0] = refState.bottom[:, 0]
        self.top[:, 0] = refState.front[:, 0]
        self.back[:, 1] = np.flip(refState.top[:, 0])
        self.bottom[:, 0] = np.flip(refState.back[:, 1])

        # Rotate main face
        self.left = np.rot90(refState.left, axes=(0, 1))
        self.update_cubeState()

    # Rotate top square to the right
    def toRight(self):
        """
        Modifies cubeState in place to rotate top face to the right
        """

        # Copy array to keep as reference
        refState = self.copy()
        # Rotate outer
        self.front[0, :] = refState.left[0, :]
        self.left[0, :] = refState.back[0, :]
        self.back[0, :] = refState.right[0, :]
        self.right[0, :] = refState.front[0, :]

        # Rotate main face
        self.top = np.rot90(refState.top, axes=(0, 1))
        self.update_cubeState()

    # Rotate top square to the left
    def toLeft(self):
        """
        Modifies cubeState in place to rotate top face to the left
        """
        # Copy array to keep as reference
        refState = self.copy()
        # Rotate outer
        self.front[0, :] = refState.right[0, :]
        self.right[0, :] = refState.back[0, :]
        self.back[0, :] = refState.left[0, :]
        self.left[0, :] = refState.front[0, :]

        # Rotate main face
        self.top = np.rot90(refState.top, axes=(1, 0))
        self.update_cubeState()
