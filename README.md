This Python script contains elements of reinforcement learning, specifically Q-learning

Components Present:
State Representation: The script represents states as locations (e.g., 'NewYork', 'NewJersey') and defines a mapping between location names and state indices.
Action Representation: Actions are implicit in the choice of the next state based on the Q-values.
Reward Function: The reward matrix R represents the immediate rewards for transitioning between states. High rewards are assigned for reaching the goal state.
Q-learning Update Rule: The Q-learning update is performed in the loop where the Q-values are updated based on the immediate reward and the maximum Q-value of the next state.
Exploration-Exploitation: The script uses an epsilon-greedy strategy for exploration-exploitation. It chooses a random action with probability epsilon and selects the action with the maximum Q-value otherwise.
