# ai/guard_agent.py

import numpy as np
import random
from collections import defaultdict, deque

class GuardAgent:
    MODES = ("PATROL", "CHASE", "SEARCH")

    def __init__(self, world, learning_rate=0.7, discount_factor=0.9):
        # Q-tables
        self.q_tables = {
            "CHASE":  defaultdict(lambda: np.zeros(4)),
            "SEARCH": defaultdict(lambda: np.zeros(4)),
        }
        self.mode = "PATROL"

        # ε-greedy
        self.epsilon       = 0.95
        self.epsilon_decay = 0.95
        self.epsilon_min   = 0.1

        self.lr    = learning_rate
        self.gamma = discount_factor

        self.world = world
        self.patrol_points = [
            (2,2),
            (2, world.size-3),
            (world.size-3, world.size-3),
            (world.size-3, 2),
        ]
        self.current_patrol_idx = 0
        self.facing = 1  # 0=up,1=down,2=left,3=right

        # Dyna-Q model
        self.model = {}

        # map memory
        self.known_grid = {}

        # stuck detection
        self.position_history = deque(maxlen=6)
        self.stuck_threshold = 4

        # chase→search grace period
        self.lost_frames = 0
        self.lost_threshold = 3

    def next_patrol_action(self, guard_pos):
        target = self.patrol_points[self.current_patrol_idx]
        if guard_pos == target:
            self.current_patrol_idx = (self.current_patrol_idx + 1) % len(self.patrol_points)
            target = self.patrol_points[self.current_patrol_idx]
        dx, dy = target[0]-guard_pos[0], target[1]-guard_pos[1]
        if abs(dx) > abs(dy):
            return 1 if dx>0 else 0
        return 3 if dy>0 else 2

    def get_action(self, state, guard_pos, last_seen_pos, player_pos, visible):
        # debug
        # print(f"[get_action] mode={self.mode}, guard={guard_pos}, player={player_pos}, visible={visible}")

        # 1) PATROL
        if self.mode == "PATROL":
            if np.random.rand() < 0.2:
                return np.random.randint(4)
            return self.next_patrol_action(guard_pos)

        # 2) CHASE
        if self.mode == "CHASE":
            goal = player_pos if visible else last_seen_pos
            if np.random.rand() < self.epsilon:
                return self._bfs_next_action(guard_pos, goal)
            return self._q_based_action(state, self.q_tables["CHASE"])

        # 3) SEARCH
        if guard_pos != last_seen_pos:
            return self._bfs_next_action(guard_pos, last_seen_pos)

        # fallback
        self.mode = "PATROL"
        return self.next_patrol_action(guard_pos)

    def update_q_values(self, state, action, reward, next_state):
        table = self.q_tables[self.mode]
        old = table[state][action]
        fut = np.max(table[next_state])
        new = (1-self.lr)*old + self.lr*(reward + self.gamma*fut)
        table[state][action] = new

        # record for Dyna-Q
        self.model.setdefault((state, action), []).append((next_state, reward))

        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        # debug
        # print(f"[{self.mode}] Q{state},{action}: {old:.2f}→{new:.2f}")

    def _q_based_action(self, state, table):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        return int(np.argmax(table[state]))

    def perform_planning_steps(self, n=5):
        for _ in range(n):
            (s,a), outcomes = random.choice(list(self.model.items()))
            s2,r = random.choice(outcomes)
            table = self.q_tables[self.mode]
            old = table[s][a]
            fut = np.max(table[s2])
            table[s][a] = (1-self.lr)*old + self.lr*(r + self.gamma*fut)

    def memorize_cell(self, pos):
        self.known_grid[pos] = self.world.grid[pos]

    def find_nearest_door_or_patrol(self, pos):
        doors = [
            p for p,val in self.known_grid.items() if val==0 and
            any(self.known_grid.get((p[0]+dx,p[1]+dy),2)==2
                for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)])
        ]
        if doors:
            return min(doors, key=lambda d: abs(d[0]-pos[0])+abs(d[1]-pos[1]))
        return self.patrol_points[self.current_patrol_idx]

    def _pos_after(self, pos, action):
        x,y = pos
        if action==0:   x=max(0,x-1)
        elif action==1: x=min(self.world.size-1,x+1)
        elif action==2: y=max(0,y-1)
        else:           y=min(self.world.size-1,y+1)
        return (x,y)

    def _bfs_next_action(self, start, goal):
        grid = self.world.grid
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        visited = {start}
        queue = deque([(start, [])])
        while queue:
            pos,path = queue.popleft()
            if pos==goal and path:
                return path[0]
            for a,(dx,dy) in enumerate(dirs):
                nxt = (pos[0]+dx, pos[1]+dy)
                if (0<=nxt[0]<self.world.size and
                    0<=nxt[1]<self.world.size and
                    grid[nxt]!=2 and nxt not in visited):
                    visited.add(nxt)
                    queue.append((nxt, path+[a]))
        return np.random.randint(4)

    @staticmethod
    def line_of_sight(start, end, grid, facing, max_range=6, fov_deg=90):
        import numpy as np
        x0,y0 = start; x1,y1 = end
        dx,dy = x1-x0, y1-y0
        dist = np.hypot(dx,dy)
        if dist>max_range or dist==0: return False
        dirs = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}
        fx,fy = dirs[facing]
        dot = (dx*fx+dy*fy)/(dist+1e-6)
        if np.degrees(np.arccos(dot))>fov_deg/2: return False
        x,y=x0,y0; sx=1 if x1>x0 else -1; sy=1 if y1>y0 else -1
        ddx,ddy=abs(dx),abs(dy)
        if ddx>ddy:
            err=ddx/2
            while x!=x1:
                if grid[x][y]==2: return False
                err-=ddy
                if err<0: y+=sy; err+=ddx
                x+=sx
        else:
            err=ddy/2
            while y!=y1:
                if grid[x][y]==2: return False
                err-=ddx
                if err<0: x+=sx; err+=ddy
                y+=sy
        return True

    def get_state(self, player_pos, guard_pos, t_since_seen, in_room):
        vis = GuardAgent.line_of_sight(guard_pos, player_pos, self.world.grid, self.facing)
        dx,dy = player_pos[0]-guard_pos[0], player_pos[1]-guard_pos[1]
        dist = np.hypot(dx,dy); bucket = min(int(dist//2),5)
        north = int(dy< -2); south = int(dy>2)
        east  = int(dx>2);  west  = int(dx< -2)
        return (
            self.mode,
            int(vis),
            bucket,
            north, south, east, west,
            min(t_since_seen,15),
            int(in_room)
        )
