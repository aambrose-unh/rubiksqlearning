import copy

# Class to define 2x2 rubiks cube
class cube:
    '''
    Initialize cube state (starts in solved state)
    Cube state is 6 x 4 matrix

               | 16 17 |
               | 16 17 |           

    | 0  1  |  | 4  5  |  | 8  9  |  | 12 13 |  | 16 17 |  | 20 21 | 
    | 2  3  |, | 6  7  |, | 10 11 |, | 14 15 |, | 18 19 |, | 22 23 |

               | 16 17 |
               | 16 17 |


    | 0   0 |  | o1 o2 |  | 0   0 |  | 0   0 |
    | 0   0 |  | o3 o4 |  | 0   0 |  | 0   0 |

    | g1 g2 |  | w1 w2 |  | b1 b2 |  | y1 y2 |
    | g3 g4 |, | w3 w4 |, | b3 b4 |, | y3 y4 |

    | 0   0 |  | r1 r2 |  | 0   0 |  | 0   0 |
    | 0   0 |  | r3 r4 |  | 0   0 |  | 0   0 |

    is equal to 




    '''


    # Randomize faces
    def randomize(self):


    # Rotate front face clockwise. 
    def clockwise(self, array):
        # Only 2x2 so only need to operate on front face. This is
        # equivalent to doing the opposite rotation on the back face.

        '''
        1           2           3           4
        | 0   0 |  | o1 o2 |  | 0   0 |  | 0   0 |
        | 0   0 |  | o3 o4 |  | 0   0 |  | 0   0 |

        | g1 g2 |  | w1 w2 |  | b1 b2 |  | y1 y2 |
        | g3 g4 |, | w3 w4 |, | b3 b4 |, | y3 y4 |

        | 0   0 |  | r1 r2 |  | 0   0 |  | 0   0 |
        | 0   0 |  | r3 r4 |  | 0   0 |  | 0   0 |

        [[[None],[Face1],[None],[None]],
        [[Face1],[Face1],[Face1],[Face1]],
        [[None],[Face1],[None],[None]],]


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
        new_array[1][1][0][1] = array[1][1][0][0]
        new_array[1][1][1][1] = array[1][1][0][1]
        new_array[1][1][1][0] = array[1][1][1][1]
        new_array[1][1][0][0] = array[1][1][1][0]

        return new_array


    # Rotate front face counterclockwise
    def counterclockwise(self):



    # Rotate lefthand square forward
    def forward(self):
        # Only 2x2 so only need to operate on left face. This is
        # equivalent to doing the opposite rotation on the right face.


    # Rotate lefthand square backward
    def backward(self):