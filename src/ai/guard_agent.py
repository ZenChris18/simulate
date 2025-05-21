import numpy as np
from collections import defaultdict

class GuardAgent:
    def __init__(self, world, learning_rate=0.7, discount_factor=0.9):
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 actions
        self.epsilon = 0.95
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.1
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.world = world
        self.memory = []

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(4)
            print(f"Exploring! Chose random action: {['Up','Down','Left','Right'][action]}")
        else:
            action = np.argmax(self.q_table[state])
            print(f"Exploiting! Best action: {['Up','Down','Left','Right'][action]} Q-value: {self.q_table[state][action]:.2f}")
        return action

    def update_q_values(self, state, action, reward, next_state):
        old_q = self.q_table[state][action]
        max_future = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_q + \
                self.learning_rate * (reward + self.discount_factor * max_future)
        self.q_table[state][action] = new_value
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        print(f"Updated Q({state}, {['Up','Down','Left','Right'][action]}) "
            f"from {old_q:.2f} to {new_value:.2f} "
            f"(Reward: {reward}, Next state: {next_state})")

    def get_state(self, player_pos, guard_pos, last_seen, in_room):
        visible = GuardAgent.line_of_sight(
            guard_pos, player_pos, self.world.grid
        )
        dx = player_pos[0] - guard_pos[0]
        dy = player_pos[1] - guard_pos[1]
        
        direction = 0  # N/S/E/W encoding
        if abs(dx) > abs(dy):
            direction = 3 if dx > 0 else 1
        else:
            direction = 2 if dy > 0 else 0
            
        distance = np.sqrt(dx**2 + dy**2)
        distance_bucket = min(int(distance // 5), 3)  # 0-3
        
        return (
            int(self.line_of_sight(guard_pos, player_pos, self.world.grid)),
            direction,
            distance_bucket,
            min(last_seen, 10),
            int(in_room)
        )

    @staticmethod
    def line_of_sight(start, end, grid):
        # Bresenham's line algorithm
        x0, y0 = start
        x1, y1 = end
        line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if grid[x][y] == 2:  # Wall check
                    return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if grid[x][y] == 2:
                    return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        return True