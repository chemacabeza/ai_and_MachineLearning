import numpy as np
import time

# Environment: A 1D track of length 6. 
# State 0 is the start, State 5 is the Goal. '---G'
N_STATES = 6
ACTIONS = ['left', 'right']

# Q-Table initialized to 0
# 6 rows for states, 2 columns for actions
q_table = np.zeros((N_STATES, len(ACTIONS)))

# Hyperparameters
EPSILON = 0.9      # 90% chance to exploit, 10% to explore
ALPHA = 0.1        # Learning rate
GAMMA = 0.9        # Discount factor for future rewards
MAX_EPISODES = 20

def choose_action(state):
    # Epsilon-greedy
    if np.random.uniform() < EPSILON:
        # Exploit (choose action with max Q-value)
        action_idx = np.argmax(q_table[state, :])
    else:
        # Explore (choose random action)
        action_idx = np.random.choice([0, 1])
    return ACTIONS[action_idx]

def get_env_feedback(state, action):
    # Move left
    if action == 'left':
        next_state = max(0, state - 1)
        reward = 0
    # Move right
    else:
        next_state = min(N_STATES - 1, state + 1)
        if next_state == N_STATES - 1:
            reward = 1  # Reached the goal!
        else:
            reward = 0
    return next_state, reward

def update_env(state, episode, step):
    env_list = ['-'] * (N_STATES - 1) + ['G']
    if state == N_STATES - 1:
        print(f"\rEpisode {episode}: Reached Goal in {step} steps!" + " " * 10)
    else:
        env_list[state] = 'A'
        print(f"\rEpisode {episode}: {''.join(env_list)}", end='')
        time.sleep(0.05) 

print("Training our Q-Learning Agent. Watch it improve!\n")
time.sleep(2)

for episode in range(1, MAX_EPISODES + 1):
    step = 0
    state = 0 # Start at the beginning
    is_terminated = False
    
    update_env(state, episode, step)
    
    while not is_terminated:
        # 1. Choose Action
        action = choose_action(state)
        action_idx = ACTIONS.index(action)
        
        # 2. Take Action, get reward and see next state
        next_state, reward = get_env_feedback(state, action)
        
        # 3. Update Q-Table
        q_predict = q_table[state, action_idx]
        if next_state != N_STATES - 1:
            q_target = reward + GAMMA * np.max(q_table[next_state, :])
        else:
            q_target = reward # Terminal state
            is_terminated = True
            
        q_table[state, action_idx] += ALPHA * (q_target - q_predict)
        
        state = next_state
        step += 1
        
        # Visuals
        update_env(state, episode, step)
        
print("\nFinal Q-Table:\n", q_table)
print("Notice how the values increase as you get closer to State 5 (Goal), naturally leading the agent there!")
