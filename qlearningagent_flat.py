import random

class Agent:
    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining = 10):
        """
        Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)

    ####################################
    #    Override These Functions      #
    ####################################
    def getQValue(self, state, action):
        """
        Should return Q(state,action)
        """
        return ('NOT DEFINED')


    def getValue(self, state):
        """
        What is the value of this state under the best action?
        Concretely, this is given by

        V(s) = max_{a in actions} Q(s,a)
        """
        return ('NOT DEFINED')

    def getPolicy(self, state):
        """
        What is the best action to take in the state. Note that because
        we might want to explore, this might not coincide with getAction
        Concretely, this is given by

        policy(s) = arg_max_{a in actions} Q(s,a)

        If many actions achieve the maximal Q-value,
        it doesn't matter which is selected.
        """
        return ('NOT DEFINED')

    def getAction(self, state):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """
        return ('NOT DEFINED')

class ReinforcementAgent(Agent):
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
    ####################################
    #    Override These Functions      #
    ####################################

    def update(self, state, action, nextState, reward):
        """
                This class will call this function, which you write, after
                observing a transition and reward
        """
        return ('NOT DEFINED')

    ####################################
    #    Read These Functions          #
    ####################################

    def getLegalActions(self,state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return self.actionFn(state)
        # return state.actionList

    def observeTransition(self, state,action,nextState,deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update(state,action,nextState,deltaReward)

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def __init__(self, actionFn = None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1):
        """
        actionFn: Function which takes a state and returns the list of legal actions

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        if actionFn == None:
            actionFn = lambda state: state.getLegalActions()
        self.actionFn = actionFn
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)

    # ################################
    # # Controls needed for Crawler  #
    # ################################
    # def setEpsilon(self, epsilon):
    #     self.epsilon = epsilon

    # def setLearningRate(self, alpha):
    #     self.alpha = alpha

    # def setDiscount(self, discount):
    #     self.discount = discount

    # def doAction(self,state,action):
    #     """
    #         Called by inherited class when
    #         an action is taken in a state
    #     """
    #     self.lastState = state
    #     self.lastAction = action

    # ###################
    # # Pacman Specific #
    # ###################
    # def observationFunction(self, state):
    #     """
    #         This is where we ended up after our last action.
    #         The simulation should somehow ensure this is called
    #     """
    #     if not self.lastState is None:
    #         reward = state.getScore() - self.lastState.getScore()
    #         self.observeTransition(self.lastState, self.lastAction, state, reward)
    #     return state

    # def registerInitialState(self, state):
    #     self.startEpisode()
    #     if self.episodesSoFar == 0:
    #         print('Beginning %d episodes of Training' % (self.numTraining))

    # def final(self, state):
    #     """
    #       Called by Pacman game at the terminal state
    #     """
    #     deltaReward = state.getScore() - self.lastState.getScore()
    #     self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
    #     self.stopEpisode()

    #     # Make sure we have this var
    #     if not 'episodeStartTime' in self.__dict__:
    #         self.episodeStartTime = time.time()
    #     if not 'lastWindowAccumRewards' in self.__dict__:
    #         self.lastWindowAccumRewards = 0.0
    #     self.lastWindowAccumRewards += state.getScore()

    #     NUM_EPS_UPDATE = 100
    #     if self.episodesSoFar % NUM_EPS_UPDATE == 0:
    #         print('Reinforcement Learning Status:')
    #         windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
    #         if self.episodesSoFar <= self.numTraining:
    #             trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
    #             print('\tCompleted %d out of %d training episodes' % (
    #                    self.episodesSoFar,self.numTraining))
    #             print('\tAverage Rewards over all training: %.2f' % (
    #                     trainAvg))
    #         else:
    #             testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
    #             print('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))
    #             print('\tAverage Rewards over testing: %.2f' % testAvg)
    #         print('\tAverage Rewards for last %d episodes: %.2f'  % (
    #                 NUM_EPS_UPDATE,windowAvg))
    #         print('\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime))
    #         self.lastWindowAccumRewards = 0.0
    #         self.episodeStartTime = time.time()

    #     if self.episodesSoFar == self.numTraining:
    #         msg = 'Training Done (turning off epsilon and alpha)'
    #         print('%s\n%s' % (msg,'-' * len(msg)))



class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # Use util.counter() to initialize Q-values
        self.qvalues = dict()

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

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
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
        return random.choice(max_action)[0]

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # With prob epsilon, take a random action
        if len(legalActions) > 0:
            # Boolean to decide if taking a random action or not
            take_random = random.random() < self.epsilon
            if take_random == True:
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)
        return action
    

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        '''
        Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
        return ('NOT DEFINED')
        '''
        # Get new q value (eq. 6.8 in Sutton and Barto)
        newQ = self.getQValue(state, action) + self.alpha * (reward + 
                self.discount*self.getValue(nextState) - self.getQValue(state, action))
        # update qvals
        self.qvalues[(state,action)] = newQ

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class ApproximateQAgent(QLearningAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def getFeatures(self, state, action):
            # feats = util.Counter()
            feats = dict()
            
            feats[(state,action)] = 1.0
            return feats

    def __init__(self):
        # self.featExtractor = getFeatures()
        # self.weights = util.Counter()
        self.weights = dict()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
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
        # return ('NOT DEFINED')

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # w_i = w_i + learning_rate * difference * f_i(s,a)
        # difference = [R + discount * max Q(s',a')] - Q(s,a)
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state,action)
        featureVector = self.getFeatures(state,action)
        for feature in featureVector:
            self.weights[feature] = self.weights[feature] + self.alpha*difference*featureVector[feature]
        # return ('NOT DEFINED')

    # def final(self, state):
    #     "Called at the end of each game."
    #     # call the super-class final method
    #     # PacmanQAgent.final(self, state)

    #     # did we finish training?
    #     if self.episodesSoFar == self.numTraining:
    #         # you might want to print your weights here for debugging
    #         "*** YOUR CODE HERE ***"
    #         pass
