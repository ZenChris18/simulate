import pygame
import numpy as np
from .world import StealthWorld
from ai.guard_agent import GuardAgent
from core.stealth_mechanics import StealthSystem

class StealthSimulation:
    def __init__(self, world_size=20):
        self.world = StealthWorld(world_size)
        self.guard = GuardAgent(self.world)
        self.player_pos = (5, 5)
        self.guard_pos = (15, 15)
        self.last_seen_pos = self.player_pos
        self.time_since_seen = 0
        self.current_level = 0
        self.tick_rate = 1.0
        self.score = 0
        self.episode = 0
        
    def hybrid_loop(self):
        self._handle_input()
        
        # Update visibility and AI
        visible = StealthSystem.calculate_visibility(
            self.guard_pos, 
            self.player_pos,
            self.world.grid,
            self.guard.facing
        )
        if visible:
            self.last_seen_pos = self.player_pos
            self.time_since_seen = 0
            self.guard.searching = False
        else:
            self.time_since_seen += 1
            # if we just lost sight, start searching
            if self.time_since_seen == 1:
                self.guard.searching = True

        # Get and execute guard action
        state = self.guard.get_state(
            self.player_pos,
            self.guard_pos,
            self.time_since_seen,
            self._in_room(self.player_pos)
        )
        action = self.guard.get_action(state, self.guard_pos, self.last_seen_pos)
        self._execute_guard_action(action)
        
        # Update Q-values
        next_state = self.guard.get_state(
            self.player_pos,
            self.guard_pos,
            self.time_since_seen,
            self._in_room(self.player_pos)
        )
        reward = self._calculate_reward(visible)
        self.guard.update_q_values(state, action, reward, next_state)
        
        # Update counters
        self.time_since_seen = 0 if visible else self.time_since_seen + 1
        
        # Control game speed (move to main loop)
        # pygame.time.wait(100)  # Optional slow down
        self.score += reward
        print(f"Episode {self.episode} | Score: {self.score:.2f} | "
              f"Visibility: {visible} | Epsilon: {self.guard.epsilon:.2f}")
        self.episode += 1

    
    def _handle_input(self):
        """Process player movement input"""
        keys = pygame.key.get_pressed()
        new_pos = list(self.player_pos)
        
        # Move player with arrow keys
        if keys[pygame.K_UP]:
            new_pos[0] = max(0, new_pos[0]-1)
        if keys[pygame.K_DOWN]:
            new_pos[0] = min(self.world.size-1, new_pos[0]+1)
        if keys[pygame.K_LEFT]:
            new_pos[1] = max(0, new_pos[1]-1)
        if keys[pygame.K_RIGHT]:
            new_pos[1] = min(self.world.size-1, new_pos[1]+1)
            
        # Check if new position is walkable
        if self.world.grid[new_pos[0]][new_pos[1]] != 2:  # Not a wall
            self.player_pos = tuple(new_pos)

    def _player_caught(self):
        """Check if guard and player are in same position"""
        return self.guard_pos == self.player_pos
    
    def _idle_too_long(self):
        return self.time_since_seen > 20  # 20 ticks without seeing player
    
    def _near_last_known(self):
        dx = abs(self.guard_pos[0] - self.last_seen_pos[0])
        dy = abs(self.guard_pos[1] - self.last_seen_pos[1])
        return dx + dy < 5
            
    def _in_room(self, pos):
        return self.world.grid[pos] == 1
    
    def _execute_guard_action(self, action):
        new_pos = list(self.guard_pos)

        self.guard.facing = action
        
        # Action mapping (same as your original: 0=up, 1=down, 2=left, 3=right)
        if action == 0:  # Up
            new_pos[0] = max(0, new_pos[0]-1)
        elif action == 1:  # Down
            new_pos[0] = min(self.world.size-1, new_pos[0]+1)
        elif action == 2:  # Left
            new_pos[1] = max(0, new_pos[1]-1)
        else:  # Right
            new_pos[1] = min(self.world.size-1, new_pos[1]+1)
        
        # Update position if not wall
        if self.world.grid[new_pos[0]][new_pos[1]] != 2:
            self.guard_pos = tuple(new_pos)
        
    def _calculate_reward(self, visible):
        if self._player_caught():
            return 100
        elif visible:
            # Add distance-based reward
            dx = abs(self.guard_pos[0] - self.player_pos[0])
            dy = abs(self.guard_pos[1] - self.player_pos[1])
            distance_reward = (20 - (dx + dy)) * 2  # Closer = higher reward
            return 20 + distance_reward
        elif self._idle_too_long():
            return -10
        else:
            return -1
        


