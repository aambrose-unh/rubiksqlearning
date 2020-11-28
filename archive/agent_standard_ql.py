import random
import time
import cube_sim

class QAgent():
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    """
      Abstract Reinforcemnt Agent: A ValueEstimationAgent
            which estimates Q-Values (as well as policies) from experience
            rather than a model

        What you need to know:
                    - The environment will call
                      observeTransition(state,action,nextState,deltaReward),
                      which will call update(state, action, nextState, deltaReward)
                      which you should override.
        - Use self.getLegalActions(state) to know which actions
                      are available in a state
    """

#################################

    # def getFeatures(self, state, action):
    #         # feats = util.Counter()
    #         feats = dict()
    #         feats[(state,action)] = 1.0
    #         return feats

    def __init__(self, numTraining=100, epsilon=0.9, alpha=0.01,
                    gamma=.9, e_decay=.00001):
        """
        alpha    - learning rate
        epsilon  - exploration rate
        e_decay  - decay rate of epsilon after each move
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.e_decay = float(e_decay)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        # self.stateCounter = 0

        # self.weights = {'inv_cpf_front': .1, 'inv_cpf_top': .1, 'inv_cpf_bottom': .1,
        #     'inv_cpf_left': .1, 'inv_cpf_right': .1, 'inv_cpf_back': .1,
        #     'inv_avg_cpf': .1, 'one_color_faces': .1}
        # self.weights = {'inv_avg_cpf': .1, 'one_color_faces': .1}
        # self.weights = {'inv_cpf_front': .1, 'inv_cpf_top': .1, 'inv_cpf_bottom': .1,
        #     'inv_cpf_left': .1, 'inv_cpf_right': .1, 'inv_cpf_back': .1, 'one_color_faces': .1}
        # self.weights = {'inv_cpf_front': .1, 'inv_cpf_top': .1, 'inv_cpf_bottom': .1,
        #     'inv_cpf_left': .1, 'inv_cpf_right': .1, 'inv_cpf_back': .1,
        #     'inv_avg_cpf': .1}

        self.qvalues = dict()

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
        feat = state.get_features()
        # if feat['one_color_faces'] == 1:
        if feat['inv_avg_cpf'] == 1:
            return list()
        return state.al

    def getFeatures(self, state, action):
            # create a copy to perform action on to get new features after action
            copy = state.copy()
            # perform action on copy
            action(copy)
            # get features
            # feats[(state,action)] = 
            # return orig_cube,state.get_features()
            return copy.get_features()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state,action) not in self.qvalues:
            self.qvalues[(state,action)] = 0.0
        return self.qvalues[(state,action)]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get list of legal actions
        actions = self.getLegalActions(state)
        # print(actions)
        # Compute Q-values for each possible action and take max
        # max Q(s',a')
        max_action = [(None,-999999)]
        for action in actions:
            # print('ON ',action)
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
        # print('Returning',action_to_return)
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
        # self.stateCounter += 1
        # if self.stateCounter % 10000 == 0:
            # self.epsilon *= .9


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

    def update(self, state, action, nextState, reward):
        """
           Q Value Update
        """
        # Get new q value (eq. 6.8 in Sutton and Barto)
        newQ = self.getQValue(state, action) + self.alpha * (reward + 
                self.discount*self.getValue(nextState) - self.getQValue(state, action))
        # update qvals
        self.qvalues[(state,action)] = newQ

    def getNextState(self, state, action):
        # # create a copy to perform action on to get new features after action
        # orig_cube = state.copy()
        # # perform action on copy
        # action()
        # return orig_cube,state
        # create a copy to perform action on to get new features after action
        copy = state.copy()
        # perform action on copy
        action(copy)
        return copy

    def train(self):
        start = time.time()
        # for number of sessions
        for sess in range(self.numTraining):
            if sess % 1 == 0:
                print('On Training Session:',sess)
            # Instantiate and randomize cube
            c = cube_sim.cube()
            # Select random initial state
            c.randomize()

            # while goal state is false
            move = 0
            prev_score = reward = 0
            while not self.isGoal(c):                
                # Reward is living cost unless terminal state then reward is 100
                if move % 10000 == 0:
                    print('\nOn Training Move:',move)
                    print('Running Time: ',time.time() - start)
                    # print('\nC to start')
                    # c.showCube()
                    c.showCube()
                    print('\nReward: {:.3f}'.format(reward))
                    print('Espilon: {:.3f}'.format(self.epsilon))

                if move > 100000000:
                    return print("Taking too long")

                action = self.getAction(c)
                # c,nextState = self.getNextState(c,action)
                nextState = self.getNextState(c,action)

                # print('\nC after getnext')
                # c.showCube()
                # print('\nNext State')
                # nextState.showCube()
                # if move % 5000 == 0:
                #     print('Cube Now')
                #     c.showCube()
                #     print('Next State')
                #     nextState.showCube()
                # Reward is difference between last state and this state
                # Score is sum of features
                # reward = (c.get_features()['inv_avg_cpf'] + 
                #     c.get_features()['one_color_faces']  - prev_score -.01)
                reward = (c.get_features()['one_color_faces']**2  - prev_score -.01)
                # reward = -.1
                if self.isGoal(nextState):
                    reward = 1000
                self.update(c, action, nextState, reward)
                # if move % 5000 == 0:
                    # print('Updated Weights',self.weights)
                prev_score = c.get_features()['one_color_faces']**2
                # print('\nNext State right before assignment')
                # nextState.showCube()
                c = nextState
                move+=1

                # print('\nC = nextState')
                # if move >= 5000:
                #     c.showCube()


    def solve(self,state,verbose = False, move_update=10, ret_moves=False):
        c = state
        print('Starting State:')
        c.showCube()
        move_list = []
        move = 0
        while self.isGoal(c) == False:
            if move % move_update == 0:
                print('On Move', move)
                if verbose == True:
                    c.showCube()
            action = self.getPolicy(c)
            c,nextState = self.getNextState(c,action)
            c = nextState
            if ret_moves == True:
                move_list.append(c)
            move+=1
            
        print('SOLVED:')
        c.showCube()
        if ret_moves == True:
            return move_list

        

a = QAgent(numTraining=10, epsilon=.9, alpha=0.1, gamma=.8, e_decay=.000005)
a.train()