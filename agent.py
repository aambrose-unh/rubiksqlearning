import random
import time
import pickle
from pathlib import Path
import numpy as np
import itertools
import cube_sim
import datetime
import copy


class ApproximateQAgent:
    """
    ApproximateQLearningAgent
    """

    def __init__(
        self, numTraining=100, epsilon=0.9, alpha=0.01, gamma=0.9, e_decay=0.00001
    ):
        """
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after
            these many episodes
        """

        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.e_decay = float(e_decay)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        # self.stateCounter = dict()
        self.episodeRewards = 0
        self.metadata = dict()

        self.weights = {
            "b": 0.1,
            "one_color_face": 0.1,
            "two_color_face": 0.1,
            "three_color_face": 0.1,
            "four_color_face": 0.1,
            "front_top": 0.1,
            "front_bottom": 0.1,
            "front_left": 0.1,
            "front_right": 0.1,
            "front_back": 0.1,
            "top_bottom": 0.1,
            "top_left": 0.1,
            "top_right": 0.1,
            "top_back": 0.1,
            "bottom_left": 0.1,
            "bottom_right": 0.1,
            "bottom_back": 0.1,
            "left_right": 0.1,
            "left_back": 0.1,
            "right_back": 0.1,
            "full_layers": 0.1,
        }

        # self.qvalues = dict()

    def getLegalActions(self, state):
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
        feat = self.getFeatures(state)
        if feat["one_color_face"] == 6:
            return list()
        return state.al

    def getFeatures(self, state, action=None):
        """
        Return features of the cube state after action
        """
        # create a copy to perform action on to get new features after action
        copy = state.copy()
        if action is not None:
            # perform action on copy
            action(copy)

        feat_vector = dict()
        feat_vector["full_layers"] = 0
        # Counts of colors per face
        tot_cpf = 0
        one_color_face = 0
        two_color_face = 0
        three_color_face = 0
        four_color_face = 0
        for face in copy.faces:
            # calc colors per face
            num_col = len(np.unique(copy.faces[face]))
            tot_cpf += num_col
            # Add 1/color for that face to feature vector
            # feat_vector[f'inv_cpf_{face}'] = (1/num_col)
            # get total single colored faces
            if num_col == 1:
                one_color_face += 1
                full_layer = 1
                for adj in copy.adjacent[face]:
                    if len(np.unique(adj)) > 1:
                        full_layer = 0
                        break
                feat_vector["full_layers"] += full_layer
            elif num_col == 2:
                two_color_face += 1
            elif num_col == 3:
                three_color_face += 1
            elif num_col == 4:
                four_color_face += 1
            else:
                return print("ERROR IN FEATURE CREATION")

        feat_vector["one_color_face"] = one_color_face
        feat_vector["two_color_face"] = two_color_face
        feat_vector["three_color_face"] = three_color_face
        feat_vector["four_color_face"] = four_color_face

        # Unique colors amongst paired faces
        for i in itertools.combinations(copy.faces, r=2):
            feat_vector[i[0] + "_" + i[1]] = 1 / len(
                np.unique(np.append(copy.faces[i[0]], copy.faces[i[1]]))
            )

        # Adjacent pair check

        feat_vector["b"] = 1

        return feat_vector

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        # Get dictionary of features
        featureVector = self.getFeatures(state, action)
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
        max_action = [(None, -999999)]
        for action in actions:
            # get current qvalue of s from taking action
            val = self.getQValue(state, action)
            # if the current action has a higher qval than the current max,
            # replace current max/action
            if (max_action[0][0] is None) or (val > max_action[0][1]):
                max_action = [(action, val)]
            # if the current action has a qval equal to the current max,
            # add the action/qval to a list to randomly choose from
            elif val == max_action[0][1]:
                max_action.append((action, val))
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
        # If there are legal actions, reset max_action to very negative number
        # and get max q-value for the set of actions
        if len(actions) > 0:
            max_action = -99999999
            for action in actions:
                Qval = self.getQValue(state, action)
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
        if self.epsilon > 0.1:
            self.epsilon = self.epsilon ** (1 + self.e_decay)
        # With prob epsilon, take a random action
        if len(legalActions) > 0:
            # Boolean to decide if taking a random action or not
            take_random = random.random() < self.epsilon
            # print(take_random)
            if take_random is True:
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)
            # print(action)
        return action

    def getRewards(self, state):
        if self.isGoal(state):
            return 1000.0
        else:
            living_tax = 1
            fl = self.getFeatures(state)["full_layers"] * 10
            ocf = self.getFeatures(state)["one_color_face"] * 6
            twcf = self.getFeatures(state)["two_color_face"] * 3
            thcf = self.getFeatures(state)["three_color_face"] * 1
            reward = fl + ocf + twcf + thcf - living_tax
            # reward = sum(i for i in self.getFeatures(state).values()) - living_tax

            return reward

    def update(self, state, action, nextState, reward):
        """
        Should update your weights based on transition
        """
        # w_i = w_i + learning_rate * difference * f_i(s,a)
        # difference = [R + discount * max Q(s',a')] - Q(s,a)
        difference = (
            reward + self.discount * self.getValue(nextState)
        ) - self.getQValue(state, action)
        featureVector = self.getFeatures(state, action)
        for feature in featureVector:
            self.weights[feature] = (
                self.weights[feature] + self.alpha * difference * featureVector[feature]
            )
        # Normalize to prevent divergence?
        max_weight = max(self.weights.values())
        for feature in self.weights:
            self.weights[feature] / max_weight

    def getNextState(self, state, action):
        # create a copy to perform action on to get new features after action
        copy = state.copy()
        # perform action on copy
        action(copy)
        return copy

    def save(self, fname, outpath=None):
        if outpath is not None:
            print(f"Saving agent to {outpath}")
            with open(outpath / f"{fname}.agent", "wb") as f:
                pickle.dump(self, f)
        else:
            print("Saving agent to working directory")
            with open(f"{fname}.agent", "wb") as f:
                pickle.dump(self, f)

    def load(self, fname, inpath=None):
        try:
            if inpath is not None:
                print(f"Loading agent from {inpath}")
                with open(inpath / f"{fname}.agent", "rb") as f:
                    return pickle.load(f)
                    # self = pickle.load(f)
            else:
                print("Loading agent from working directory")
                with open(f"{fname}.agent", "rb") as f:
                    return pickle.load(f)
        except:
            print("File Not Found")

    def train(self, save_prefix="mdl_", moves_per_ep=1000):
        print("Starting Training")
        start = time.time()
        if len(self.metadata) > 1:
            start_sess = max([int(i[3:]) for i in self.metadata.keys()])
        else:
            start_sess = 1
        # for number of sessions
        for sess in range(start_sess, start_sess + self.numTraining + 1):
            self.episodeRewards = 0
            train_start = time.time()
            # if sess % 1000 == 0:
            #     print("On Training Session:", sess)
            #     print("Total Running Time: ", time.time() - start)
            # Instantiate and randomize cube
            c = cube_sim.cube()
            # Select random initial state
            c.randomize()

            # while goal state is false
            move = 0
            # prev_score = -.1
            reward = -0.1
            while (not self.isGoal(c)) and (move < moves_per_ep):
                # if move % 10000 == 0:
                #     print("\nTraining Session {}\nTraining Move: {}".format(sess, move))
                #     print("Training Running Time: ", time.time() - train_start)
                #     print("Total Running Time: ", time.time() - start)
                #     c.showCube()
                #     print("\nReward: {:.3f}".format(reward))
                #     print("Weights: {}".format(self.weights))
                #     print("Espilon: {:.3f}".format(self.epsilon))

                action = self.getAction(c)
                nextState = self.getNextState(c, action)
                reward = self.getRewards(c)
                self.update(c, action, nextState, reward)
                c = nextState
                self.episodeRewards += reward
                if self.isGoal(c):
                    print("\nGOAL !!!!")
                    print("\nTraining Session {}\nTraining Move: {}".format(sess, move))
                    print(datetime.datetime.now())
                    # if nextState is goal add the rewards from the goal state
                    self.episodeRewards += self.getRewards(c)
                    c.showCube()
                move += 1

            # Save model after each completed training episode
            if sess % 1000 == 0:
                print("\nOn Training Session:", sess)
                print("Total Running Time: ", time.time() - start)
                print(datetime.datetime.now())
                print("Epsilon", self.epsilon)
                print("Saving model")
                self.save(
                    fname=f"{save_prefix}_episode_{sess}", outpath=Path("./output")
                )
            self.metadata[f"ep_{sess}"] = {
                "MovesToGoal": move - 1,
                "EpisodeRewards": self.episodeRewards,
                "Weights": self.weights,
                "TotalRunTime": time.time() - start,
                "EpisodeRunTime": time.time() - train_start,
            }

    def solve(
        self, state, max_moves=10000, verbose=False, move_update=5000, ret_moves=False
    ):
        c = state.copy()
        print("Starting State:")
        start = time.time()
        c.showCube()
        move_list = []
        solved = 0
        move = 0
        while (self.isGoal(c) is False) and (move < max_moves):
            if move % move_update == 0:
                print("\nOn Move", move)
                print("Run time:", time.time() - start)
                if verbose is True:
                    print("verbose = true")
                    c.showCube()
            action = self.getPolicy(c)
            nextState = self.getNextState(c, action)
            c = nextState
            if ret_moves is True:
                move_list.append(action)
            move += 1
        runtime = time.time() - start
        if self.isGoal(c):
            solved = 1
            print("SOLVED:")
            print("\nOn Move", move)
        else:
            print("Hit max moves:", max_moves)
        print("Total Run time:", runtime)
        print(datetime.datetime.now())
        # c.showCube()
        to_ret = move, runtime, solved, c
        if ret_moves is True:
            return move_list, to_ret
        return to_ret


class evaluator:
    def __init__(self, agent, numIterations=100):
        self.numIterations = int(numIterations)
        self.eval_data = dict()
        # self.agent = copy.deepcopy(agent)
        self.agent = agent

    def run(self):
        # For num of iterations, solve cubes and record info
        print("Begin Evaluation")
        for iter in range(self.numIterations):
            print("Starting iteration", iter)
            c = cube_sim.cube()
            c.randomize()
            startCube = c.copy()
            moves, runtime, solved, endCube = self.agent.solve(
                c, max_moves=5000, verbose=False, move_update=6000
            )
            self.eval_data[f"iter{iter}"] = {
                "MovesInIter": moves,
                "TimeToRun": runtime,
                "Solved": solved,
                "StartCube": startCube,
                "EndCube": endCube,
            }

    def save(self, fname, outpath=None):
        if outpath is not None:
            print(f"Saving eval to {outpath}")
            with open(outpath / f"{fname}.eval", "wb") as f:
                pickle.dump(self, f)
        else:
            print("Saving eval to working directory")
            with open(f"{fname}.eval", "wb") as f:
                pickle.dump(self, f)

    def load(self, fname, inpath=None):
        try:
            if inpath is not None:
                print(f"Loading eval from {inpath}")
                with open(inpath / f"{fname}.eval", "rb") as f:
                    return pickle.load(f)
                    # self = pickle.load(f)
            else:
                print("Loading eval from working directory")
                with open(f"{fname}.eval", "rb") as f:
                    return pickle.load(f)
        except:
            print("File Not Found")


# a = ApproximateQAgent(numTraining=10, epsilon=.75, alpha=0.1, gamma=.9, e_decay=.00001)
# a.train()
