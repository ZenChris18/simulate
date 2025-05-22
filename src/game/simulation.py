import pygame, random
from .world import StealthWorld
from ai.guard_agent import GuardAgent
from core.stealth_mechanics import StealthSystem

class StealthSimulation:
    def __init__(self, world_size=20):
        self.world         = StealthWorld(world_size)
        self.guard         = GuardAgent(self.world)
        self.player_pos    = (5,5)
        self.guard_pos     = (15,15)
        self.last_seen_pos = self.player_pos
        self.time_since_seen = 0
        self.score         = 0
        self.episode       = 0

    def hybrid_loop(self):
        # 0) build map memory
        self.guard.memorize_cell(self.guard_pos)

        # 1) player input
        self._handle_input()

        # 2) visibility
        visible = StealthSystem.calculate_visibility(
            self.guard_pos, self.player_pos, self.world.grid, self.guard.facing
        )

        # 3) mode transition
        if visible:
            self.guard.mode = "CHASE"
            self.guard.lost_frames = 0
            self.last_seen_pos = self.player_pos
            self.time_since_seen = 0

        elif self.guard.mode=="CHASE":
            self.guard.lost_frames += 1
            if self.guard.lost_frames>self.guard.lost_threshold:
                self.guard.mode="SEARCH"
                self.time_since_seen=1

        elif self.guard.mode=="SEARCH":
            self.time_since_seen +=1

        # 4) state & action
        in_room = (self.world.grid[self.player_pos]==1)
        state    = self.guard.get_state(
            self.player_pos,
            self.guard_pos,
            self.time_since_seen,
            in_room
        )
        action   = self.guard.get_action(
            state,
            self.guard_pos,
            self.last_seen_pos,
            self.player_pos,
            visible
        )
        # debug
        # print(f"[Sim] action={action}")

        # 5) move guard
        self.guard.facing = action
        new_pos = self._pos_after_action(self.guard_pos, action)
        # wall check
        if self.world.grid[new_pos]==2:
            valid = [a for a in (0,1,2,3)
                     if self.world.grid[self._pos_after_action(self.guard_pos,a)]!=2]
            action= random.choice(valid) if valid else action
            self.guard.facing = action
            new_pos= self._pos_after_action(self.guard_pos,action)
        self.guard_pos = new_pos

        # 6) reward + learning
        reward = self._calculate_reward(visible)
        if self.guard.mode in ("CHASE","SEARCH"):
            next_state = self.guard.get_state(
                self.player_pos,
                self.guard_pos,
                self.time_since_seen,
                in_room
            )
            self.guard.update_q_values(state, action, reward, next_state)
            self.guard.perform_planning_steps(3)

        # 7) stuck escape
        self.guard.position_history.append(self.guard_pos)
        # Only check if we have enough history and positions are stagnant
        if (len(self.guard.position_history) >= self.guard.position_history.maxlen and 
            len(set(self.guard.position_history)) < self.guard.stuck_threshold):
            tgt = self.guard.find_nearest_door_or_patrol(self.guard_pos)
            a = self.guard._bfs_next_action(self.guard_pos, tgt)
            self.guard_pos = self._pos_after_action(self.guard_pos, a)
            self.guard.position_history.clear()

        # 8) bookkeeping
        if not visible:
            self.time_since_seen +=1
        self.score += reward
        print(f"Ep{self.episode} | mode={self.guard.mode} | score={self.score:.1f} | ε={self.guard.epsilon:.2f}")
        self.episode +=1

    def _handle_input(self):
        keys = pygame.key.get_pressed()
        x,y = self.player_pos
        if keys[pygame.K_UP]:
            x = max(0, x-1)
        if keys[pygame.K_DOWN]:
            x = min(self.world.size-1, x+1)
        if keys[pygame.K_LEFT]:
            y = max(0, y-1)
        if keys[pygame.K_RIGHT]:
            y = min(self.world.size-1, y+1)
        if self.world.grid[x][y] != 2:
            self.player_pos = (x,y)

    def _pos_after_action(self, pos, action):
        x,y = pos
        if action==0:   x = max(0, x-1)
        elif action==1: x = min(self.world.size-1, x+1)
        elif action==2: y = max(0, y-1)
        else:           y = min(self.world.size-1, y+1)
        return (x,y)

    def _execute_guard_action(self, action):
        old = self.guard_pos
        self.guard.facing = action
        new = self._pos_after_action(old, action)

        if self.world.grid[new[0]][new[1]] == 2:
            # wall: pick a valid random move
            valid = [
                a for a in (0,1,2,3)
                if self.world.grid[self._pos_after_action(old,a)[0]]
                                   [self._pos_after_action(old,a)[1]] != 2
            ]
            if valid:
                action = random.choice(valid)
                new = self._pos_after_action(old, action)
                self.guard.facing = action

        self.guard_pos = new

    def _player_caught(self):
        return self.guard_pos == self.player_pos

    def _idle_too_long(self):
        return self.time_since_seen > 20

    def _in_room(self, pos):
        return self.world.grid[pos] == 1

    def _calculate_reward(self, visible):
        if self._player_caught():
            return 100
        if self.guard.mode == "CHASE":
            # reward for closing in
            old_dist = abs(self.guard_pos[0] - self.player_pos[0]) + \
                       abs(self.guard_pos[1] - self.player_pos[1])
            # after action update, so smaller distance → positive
            # here we approximate current dist as new
            return (20 - old_dist) * 2
        if self.guard.mode == "SEARCH":
            # reward for moving toward last seen
            dx = abs(self.guard_pos[0] - self.last_seen_pos[0])
            dy = abs(self.guard_pos[1] - self.last_seen_pos[1])
            return ( (self.world.size - (dx+dy)) * 1.5 ) - 1
        # PATROL gets tiny per-step bonus
        return 0.1
