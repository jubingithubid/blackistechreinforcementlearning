import numpy as np
import random

# Parameters
gamma = 0.85  # Discount factor
alpha = 0.9   # Learning rate
initial_epsilon = 1.0  # Starting exploration factor
decay = 0.999  # Decay rate for epsilon
min_epsilon = 0.1  # Minimum value for epsilon

# States and actions
location_to_state = {
    'NewYork': 0, 'NewJersey': 1, 'Pennsylvania': 2, 'Delaware': 3, 'Maryland': 4,
    'Baltimore': 5, 'Washington DC': 6, 'Virginia': 7, 'North Carolina': 8,
    'South Carolina': 9, 'Georgia': 10, 'Florida': 11
}
state_to_location = {state: location for location, state in location_to_state.items()}

# Reward matrix
R = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # NY
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # NJ
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # PA
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # DE
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # MD
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # BM
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # DC
    [0, 0, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1],  # VA
    [0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0],  # NC
    [0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0],  # SC
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 1],  # GA
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]   # FL
])

def route(starting_location, ending_location):
    start_state = location_to_state[starting_location]
    end_state = location_to_state[ending_location]
    R_new = np.copy(R)
    R_new[end_state, end_state] = 1000  # High reward for reaching the goal

    Q = np.zeros([12, 12])
    epsilon = initial_epsilon
    for i in range(50000):  # Increased iterations for more thorough learning
        current_state = np.random.randint(0, 12)
        if random.uniform(0, 1) < epsilon:
            playable_actions = np.where(R_new[current_state] > 0)[0]
            next_state = np.random.choice(playable_actions)
        else:
            next_state = np.argmax(Q[current_state])

        # Q-value update
        TD = R_new[current_state, next_state] + gamma * np.max(Q[next_state]) - Q[current_state, next_state]
        Q[current_state, next_state] += alpha * TD
        epsilon = max(min_epsilon, epsilon * decay)

    # Generate the route
    route = [starting_location]
    next_location = starting_location
    while next_location != ending_location:
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location

    return route

final_route = route('North Carolina', 'NewYork')
print('Route:')
print(final_route)
