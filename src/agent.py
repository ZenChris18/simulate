import numpy as np

class Explorer:
    def __init__(self, grid_size):
        self.q_table = np.zeros((grid_size, grid_size, 4))  # Grid states × 4 actions
        self.epsilon = 0.9  # Exploration rate
        self.epsilon_decay = 0.97
        self.epsilon_min = 0.1
    
    def get_action(self, state):
        """Chooses action: 0=up, 1=down, 2=left, 3=right."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(4)  # Explore
        else:
            x, y = state
            return np.argmax(self.q_table[x][y])  # Exploit
    
    def update(self, state, action, reward, next_state):
        """Q-learning update rule."""
        x, y = state
        next_x, next_y = next_state
        old_value = self.q_table[x][y][action]
        max_future = np.max(self.q_table[next_x][next_y])
        new_value = old_value + 0.5 * (reward + 0.95 * max_future - old_value)  # α=0.5, γ=0.95 formula for the algorithm
        self.q_table[x][y][action] = new_value
