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
        self.patrol_points = [(2,2), (2, world.size-3), (world.size-3, world.size-3), (world.size-3, 2)]
        self.current_patrol_idx = 0
        self.facing = 1
        self.searching = False
        self.memory = []

    def next_patrol_action(self, guard_pos):
        target = self.patrol_points[self.current_patrol_idx]
        if guard_pos == target:
            self.current_patrol_idx = (self.current_patrol_idx + 1) % len(self.patrol_points)
            target = self.patrol_points[self.current_patrol_idx]
        # simple greedy towards target:
        dx = target[0] - guard_pos[0]
        dy = target[1] - guard_pos[1]
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 0  # down / up
        else:
            return 3 if dy > 0 else 2  # right / left

    def get_action(self, state, guard_pos, last_seen_pos):
        visible = state[0]
        if visible:  # If player is visible
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.9)
        
        # 2) If searching (lost sight but have last_seen):
        if self.searching and guard_pos != last_seen_pos:
            return self.move_toward(guard_pos, last_seen_pos)

        # 3) Otherwise, maybe patrol:
        if np.random.rand() < 0.2:
            return self.next_patrol_action(guard_pos)

        # 4) Fallback: Q-based
        return self._q_based_action(state)
    
    def move_toward(self, guard_pos, target):
        gx,gy = guard_pos
        tx,ty = target
        dx, dy = tx-gx, ty-gy
        if abs(dx) > abs(dy):
            return 1 if dx>0 else 0
        else:
            return 3 if dy>0 else 2


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
        visible = GuardAgent.line_of_sight(guard_pos, player_pos, self.world.grid, self.facing)
        dx = player_pos[0] - guard_pos[0]
        dy = player_pos[1] - guard_pos[1]
        
        # More granular distance buckets
        distance = np.sqrt(dx**2 + dy**2)
        distance_bucket = min(int(distance // 2), 5)  # 0-5 buckets
        
        # Relative position flags
        north = 1 if dy < -2 else 0
        south = 1 if dy > 2 else 0
        east = 1 if dx > 2 else 0
        west = 1 if dx < -2 else 0
        
        return (
            int(visible),
            distance_bucket,
            north, south, east, west,
            min(last_seen, 15),
            int(in_room)
        )

    @staticmethod
    def line_of_sight(start, end, grid, facing, max_range=8, fov_deg=90):
        import numpy as np

        x0,y0 = start; x1,y1 = end
        dx, dy = x1-x0, y1-y0
        dist = np.hypot(dx, dy)
        # 1) Range check
        if dist > max_range or dist == 0:
            return False

        # 2) Cone check
        # facing → unit vector
        dirs = { 0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1) }
        fx, fy = dirs[facing]
        dot = (dx*fx + dy*fy) / (dist + 1e-6)
        angle = np.degrees(np.arccos(dot))
        if angle > fov_deg/2:
            return False

        # 3) Bresenham through grid walls
        x,y = x0,y0
        sx = 1 if x1>x0 else -1
        sy = 1 if y1>y0 else -1
        ddx, ddy = abs(dx), abs(dy)
        if ddx > ddy:
            err = ddx/2
            while x!=x1:
                if grid[x][y]==2: return False
                err -= ddy
                if err<0:
                    y += sy; err += ddx
                x += sx
        else:
            err = ddy/2
            while y!=y1:
                if grid[x][y]==2: return False
                err -= ddx
                if err<0:
                    x += sx; err += ddy
                y += sy
        return True

    def _q_based_action(self, state):
        # classic ε-greedy on self.q_table
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        return int(np.argmax(self.q_table[state]))
