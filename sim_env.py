import agent
import cube_sim

# Instantiate cube
c = cube_sim.cube()
# Select random initial state
c.randomize()


# Instantiate Q Learning Agent
a = agent.ApproximateQAgent()

# While goal state hasnt been reached, 