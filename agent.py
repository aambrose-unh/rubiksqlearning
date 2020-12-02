import random
import time
import pickle
from pathlib import Path
import numpy as np
import itertools
import cube_sim

class ApproximateQAgent():
    """
       ApproximateQLearningAgent
    """
    def __init__(self, numTraining=100, epsilon=0.9, alpha=0.01, gamma=.9, e_decay=.00001):
        """
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """

        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.e_decay = float(e_decay)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        # self.stateCounter = dict()
        self.metadata = dict()
        self.prev_moves = [0]*6*6

        self.weights = dict(zip([f'p_{i}_{m}'for i in range(6,0,-1) for m in ['cw','ccw','f','b','l','r']],[.1]*6*6))
        # self.weights = {'inv_cpf_front': 0.1, 'inv_cpf_top': 0.1, 'inv_cpf_bottom': 0.1, 'inv_cpf_left': 0.1, 'inv_cpf_right': 0.1, 'inv_cpf_back': 0.1, 'inv_avg_cpf': 0.1, 'front_top': 0.1, 'front_bottom': 0.1, 'front_left': 0.1, 'front_right': 0.1, 'front_back': 0.1, 'top_bottom': 0.1, 'top_left': 0.1, 'top_right': 0.1, 'top_back': 0.1, 'bottom_left': 0.1, 'bottom_right': 0.1, 'bottom_back': 0.1, 'left_right': 0.1, 'left_back': 0.1, 'right_back': 0.1, 'one_color_faces': .1}
        # self.weights = {'inv_avg_cpf': 0.1, 'one_color_faces': 0.1, 'front_top': 0.1, 'front_bottom': 0.1, 'front_left': 0.1, 'front_right': 0.1, 'front_back': 0.1, 'top_bottom': 0.1, 'top_left': 0.1, 'top_right': 0.1, 'top_back': 0.1, 'bottom_left': 0.1, 'bottom_right': 0.1, 'bottom_back': 0.1, 'left_right': 0.1, 'left_back': 0.1, 'right_back': 0.1}
        # self.weights = {'front_top': 0.1, 'front_bottom': 0.1, 'front_left': 0.1, 'front_right': 0.1, 'front_back': 0.1, 'top_bottom': 0.1, 'top_left': 0.1, 'top_right': 0.1, 'top_back': 0.1, 'bottom_left': 0.1, 'bottom_right': 0.1, 'bottom_back': 0.1, 'left_right': 0.1, 'left_back': 0.1, 'right_back': 0.1}
        # ALL FEATURES
        # self.weights = {'inv_cpf_front': 0.1, 'inv_cpf_top': 0.1, 'inv_cpf_bottom': 0.1, 'inv_cpf_left': 0.1, 'inv_cpf_right': 0.1, 'inv_cpf_back': 0.1, 'inv_avg_cpf': 0.1, 'one_color_faces': 0.1, 'front_top': 0.1, 'front_bottom': 0.1, 'front_left': 0.1, 'front_right': 0.1, 'front_back': 0.1, 'top_bottom': 0.1, 'top_left': 0.1, 'top_right': 0.1, 'top_back': 0.1, 'bottom_left': 0.1, 'bottom_right': 0.1, 'bottom_back': 0.1, 'left_right': 0.1, 'left_back': 0.1, 'right_back': 0.1, 'flt': 0.1, 'frt': 0.1, 'flb': 0.1, 'frb': 0.1, 'blt': 0.1, 'brt': 0.1, 'blb': 0.1, 'brb': 0.1}

        # self.qvalues = dict()

    def getLegalActions(self,state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return self.getActions(state)

    def isGoal(self, state):
        if len(self.getActions(state)) == 0:
            return True
        return False

    def getActions(self, state):
        # feat = self.getFeatures(state)
        # if feat['one_color_faces'] == 1:
        tot_cpf = 0
        for face in state.faces:
            # calc colors per face
            num_col = len(np.unique(state.faces[face]))
            tot_cpf += num_col
            # Add 1/color for that face to feature vector
            # feat_vector[f'inv_cpf_{face}'] = (1/num_col)
            # get total single colored faces
            # if num_col == 1:
            #     one_color_face += 1
        # if feat['inv_avg_cpf'] == 1:
        if tot_cpf == 6:
            return list()
        return state.al

    def getFeatures(self, state, action=None):
        '''
        Return features of the cube state after action
        '''
        moves = self.prev_moves.copy()
        # create a copy to perform action on to get new features after action
        copy = state.copy()
        if action != None:
            # perform action on copy
            action(copy)
            newmove = copy.al.index(action)
            toadd = [0]*6
            toadd[newmove] = 1
            moves.extend(toadd)
            moves = moves[6:]
            feat_vector = dict(zip([f'p_{i}_{m}'for i in range(6,0,-1) for m in ['cw','ccw','f','b','l','r']],moves))
        else:
            feat_vector = dict(zip([f'p_{i}_{m}'for i in range(6,0,-1) for m in ['cw','ccw','f','b','l','r']],self.prev_moves))
     
        # Maybe add interaction features for each past move and inv_avg_cpf to say something
        # about the states during each move?

        # Maybe add interaction between move and each other move?

        # tot_cpf = 0
        # one_color_face = 0
        # # for face in copy.faces:
        #     # calc colors per face
        #     num_col = len(np.unique(copy.faces[face]))
        #     tot_cpf += num_col
        #     # Add 1/color for that face to feature vector
        #     # feat_vector[f'inv_cpf_{face}'] = (1/num_col)
        #     # get total single colored faces
        #     if num_col == 1:
        #         one_color_face += 1

        # # Unique colors amongst paired faces
        # for i in itertools.combinations(copy.faces,r=2):
        #     feat_vector[i[0]+'_'+i[1]] = 1/len(np.unique(np.append(copy.faces[i[0]],copy.faces[i[1]])))

        # # Add 1 / average colors per face
        # feat_vector['inv_avg_cpf'] = (1/(tot_cpf/6))
        # # Add number of single colored faces (divided by 6 to keep same scale as other features)
        # feat_vector['one_color_faces'] = (one_color_face/6)

        # # Interaction Features
        # for i in itertools.combinations(self.faces,r=2):
        #     num_col0 = 1/len(np.unique(self.faces[i[0]]))
        #     num_col1 = 1/len(np.unique(self.faces[i[1]]))
        #     feat_vector[i[0]+'_'+i[1]] = num_col0*num_col1

        # Maybe check for patterns of previous moves????
        
        # UPDATE BASED ON P1 past actions (attribute and in update function)

        return feat_vector

        

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # Get dictionary of features
        featureVector = self.getFeatures(state,action)
        # Get dictionary mapping features to weights
        weightVector = self.getWeights()
        sum = 0
        # For each feature in the feature vector, multiply by the
          # weights and get the sum as the approximate q-value
        for feature in featureVector:
            sum += weightVector[feature] * featureVector[feature]
        return sum

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get list of legal actions
        actions = self.getLegalActions(state)
        # Compute Q-values for each possible action and take max
        # max Q(s',a')
        max_action = [(None,-999999)]
        for action in actions:
            # get current qvalue of s from taking action
            val = self.getQValue(state,action)
            # if the current action has a higher qval than the current max,
              # replace current max/action
            if (max_action[0][0] == None) or (val > max_action[0][1]):
                max_action = [(action,val)]
            # if the current action has a qval equal to the current max,
              # add the action/qval to a list to randomly choose from
            elif (val == max_action[0][1]):
                max_action.append((action,val))
        # if more than one action results in max qvalue - choose one randomly
        action_to_return = random.choice(max_action)[0]
        return action_to_return

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # Initialize to zero - return zero if no legal actions
        max_action = 0
        # Get legal actions
        actions = self.getLegalActions(state)
        # If there are legal actions, reset max_action to very negative number and get
            # max q-value for the set of actions
        if (len(actions) > 0):
            max_action = -99999999
            for action in actions:
                Qval = self.getQValue(state,action)
                if Qval > max_action:
                    max_action = Qval
        return max_action

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)
    
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        # Epsilon decay
        if self.epsilon > .1:
            self.epsilon = self.epsilon ** (1+self.e_decay)
        # With prob epsilon, take a random action
        if len(legalActions) > 0:
            # Boolean to decide if taking a random action or not
            take_random = random.random() < self.epsilon
            # print(take_random)
            if take_random == True:
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)
            # print(action)
        return action

    def getRewards(self,state):
        tot_cpf = 0
        one_color_face = 0
        for face in state.faces:
            # calc colors per face
            num_col = len(np.unique(state.faces[face]))
            tot_cpf += num_col
            # Add 1/color for that face to feature vector
            # feat_vector[f'inv_cpf_{face}'] = (1/num_col)
            # get total single colored faces
            if num_col == 1:
                one_color_face += 1

        # # Add 1 / average colors per face
        # feat_vector['inv_avg_cpf'] = (1/(tot_cpf/6))
        # # Add number of single colored faces (divided by 6 to keep same scale as other features)
        # feat_vector['one_color_faces'] = (one_color_face/6)

        # MAYBE BUMP CPF IMPORTANCE?

        # Maybe add windowed average of these to the features and/or reward?

        return (1/(tot_cpf/6)) + (one_color_face/6) - .1

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # w_i = w_i + learning_rate * difference * f_i(s,a)
        # difference = [R + discount * max Q(s',a')] - Q(s,a)
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state,action)
        featureVector = self.getFeatures(state,action)
        for feature in featureVector:
            self.weights[feature] = self.weights[feature] + self.alpha*difference*featureVector[feature]
        # Normalize to prevent divergence?
        max_weight = max(self.weights.values())
        for feature in self.weights: 
            self.weights[feature] / max_weight

        # Update past actions list
        self.prev_moves = [i for i in featureVector.values()]


    def getNextState(self, state, action):
        # create a copy to perform action on to get new features after action
        copy = state.copy()
        # perform action on copy
        action(copy)
        return copy

    def save(self,fname,outpath=None):
        if outpath != None:
            print(f'Saving agent to {outpath}')
            with open(outpath / f'{fname}.agent','wb') as f:
                pickle.dump(self,f)
        else:
            print(f'Saving agent to working directory')
            with open(f'{fname}.agent','wb') as f:
                pickle.dump(self,f)

    def load(self,fname,inpath=None):
        try:
            if inpath != None:
                print(f'Loading agent from {inpath}')
                with open(inpath / f'{fname}.agent','rb') as f:
                    return pickle.load(f)
                    # self = pickle.load(f)
            else:
                print(f'Loading agent from working directory')
                with open(f'{fname}.agent','rb') as f:
                    return pickle.load(f)
        except:
            print('File Not Found')


    def train(self,save_prefix='mdl_'):
        start = time.time()
        # for number of sessions
        for sess in range(self.numTraining):
            train_start = time.time()
            if sess % 1 == 0:
                print('On Training Session:',sess)
            # Instantiate and randomize cube
            c = cube_sim.cube()
            # Select random initial state
            c.randomize()

            # while goal state is false
            move = 0
            # prev_score = -.1
            reward=-.1
            while not self.isGoal(c):                
                # Reward is living cost unless terminal state then reward is 100
                if move % 10000 == 0:                
                    print('\nTraining Session {}\nTraining Move: {}'.format(sess,move))
                    print('Training Running Time: ',time.time() - train_start)
                    print('Total Running Time: ',time.time() - start)
                    c.showCube()
                    print('\nReward: {:.3f}'.format(reward))
                    print('Weights: {}'.format(self.weights))
                    print('Espilon: {:.3f}'.format(self.epsilon))
                    # print('Number of States visited: {}\n'.format(len(self.stateCounter)))


                if move > 10000000:
                    self.save(fname=f'{save_prefix}_froze_episode_{sess}',outpath=Path('./output'))
                    return print("Taking too long - moves =",move)

                action = self.getAction(c)
                # c,nextState = self.getNextState(c,action)
                nextState = self.getNextState(c,action)

                # reward = c.get_features()['one_color_faces'] - prev_score - .1
                # reward = c.get_features()['inv_avg_cpf']**2 - prev_score - .1
                # reward = self.getFeatures(c)['inv_avg_cpf'] + self.getFeatures(c)['one_color_faces'] - .1
                reward = self.getRewards(c)
                # reward = -.1
                if self.isGoal(nextState):
                    print('\nGOAL HERE!!!!')
                    print('\nTraining Session {}\nTraining Move: {}'.format(sess,move))
                    nextState.showCube()
                    reward = 100
                self.update(c, action, nextState, reward)
                # prev_score = c.get_features()['one_color_faces'] + c.get_features()['inv_avg_cpf']
                # prev_score = c.get_features()['one_color_faces']
                # prev_score = c.get_features()['inv_avg_cpf']**2
                # print('\nNext State right before assignment')
                # nextState.showCube()
                c = nextState
                move+=1
            
            # Save model after each completed training episode
            print('Saving model')
            self.metadata[f'ep_{sess}'] = {'MovesToGoal':move-1,'TotalRunTime':time.time() - start,'EpisodeRunTime':time.time() - train_start}
            self.save(fname=f'{save_prefix}_episode_{sess}',outpath=Path('./output'))


    def solve(self,state,verbose = False, move_update=1000, ret_moves=False):
        c = state.copy()
        print('Starting State:')
        start = time.time()
        c.showCube()
        move_list = []
        move = 0
        while self.isGoal(c) == False:
            if move % move_update == 0:
                print('\nOn Move', move)
                print('Run time:',time.time() - start)
                if verbose == True:
                    print("verbose = true")
                    c.showCube()
            action = self.getPolicy(c)
            nextState = self.getNextState(c,action)
            c = nextState
            if ret_moves == True:
                move_list.append(action)
            move+=1
            
        print('SOLVED:')
        print('\nOn Move', move)
        print('Total Run time:',time.time() - start)
        c.showCube()
        if ret_moves == True:
            return move_list

        

# a = ApproximateQAgent(numTraining=10, epsilon=.75, alpha=0.1, gamma=.9, e_decay=.00001)
# a.train()