# rubiksqlearning
Reinforcement learning used to solve 2x2 rubiks cube

# Background Info
The objectives for this project were:
1. Make a python module to simulate a rubik's cube
1. Train a reinforcement learning agent using feature based learning (approximate q-learning)
1. Evaluate the reinforcement learning agent as compared to an agent that randomly selects moves to make

# Basic Project Information
The environment was created using pipenv. With pipenv and python 3.9 installed, this repo can be downloaded and the command `pipenv shell` can be ran in the console within the project directory. This sets that console to run inside the virtual environment.


# Training

1. Instantiate training agent:
    - `a = agent.ApproximateQAgent(numTraining=4000, epsilon=.9, alpha=0.005, gamma=.9, e_decay=.000002)`
        - numTraining = the number of training episodes to run
        - epsilon = the starting probability of choosing a random action
        - alpha = learning rate
        - e_decay = rate of decay for epsilon during training
1. Train agent
    - `a.train(save_prefix='12_05_v1')`
        - save_prefix = the prefix to put at the beginning of the saved filename to differentiate the file

Useful methods:
 - `agent.save(fname, outpath)`
    - Saves current state of agent
- `agent.load(fname, inpath)`
    - Loads a saved agent
- `agent.solve(state, max_moves=10000, verbose=False, move_update=5000, ret_moves=False)`
    - Will solve a cube using the agent's policy

# Evaluating

1. Instantiate evaluator with the agent to be evaluated
    - `evaluator = agent.evaluator(a,numIterations=1000)`
        - a = agent
        - numIterations = number of iterations to run the solver
1. Run the evaluator
    - `evaluator.run(save_prefix='eval_',max_moves=10000)`
        - save_prefix = the prefix to put at the beginning of the saved filename to differentiate the file
        - max_moves = the max number of moves to allow the agent to solve a cube before claiming it is not solved and restarting
    
    
