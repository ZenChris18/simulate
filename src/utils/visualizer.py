import pygame
from ai.guard_agent import GuardAgent
import numpy as np

class StealthVisualizer:
    COLORS = {
        'floor': (40, 40, 40),
        'wall': (80, 80, 80),
        'room': (120, 80, 50),
        'guard': (200, 0, 0),
        'player': (0, 0, 200),
        'fov': (255, 0, 0, 50)
    }

    @staticmethod
    def render(screen, world, player_pos, guard_pos):
        screen_width = screen.get_width()
        cell_size = screen_width // world.size
        screen.fill((0, 0, 0))
        
        for x in range(world.size):
            for y in range(world.size):
                color = StealthVisualizer.COLORS['floor']
                if world.grid[x][y] == 1:
                    color = StealthVisualizer.COLORS['room']
                elif world.grid[x][y] == 2:
                    color = StealthVisualizer.COLORS['wall']
                
                pygame.draw.rect(screen, color,
                               (y*cell_size, x*cell_size,
                                cell_size, cell_size))
        
        # Draw guard
        pygame.draw.circle(screen, StealthVisualizer.COLORS['guard'],
                          (guard_pos[1]*cell_size + cell_size//2,
                           guard_pos[0]*cell_size + cell_size//2),
                          cell_size//3)
        
        if GuardAgent.line_of_sight(guard_pos, player_pos, world.grid):
            fov_color = (255, 0, 0, 50)
            points = [
                (guard_pos[1]*cell_size + cell_size//2, guard_pos[0]*cell_size + cell_size//2),
                (player_pos[1]*cell_size + cell_size//2, player_pos[0]*cell_size + cell_size//2),
            ]
        
        # Draw player
        pygame.draw.circle(screen, StealthVisualizer.COLORS['player'],
                          (player_pos[1]*cell_size + cell_size//2,
                           player_pos[0]*cell_size + cell_size//2),
                          cell_size//3)
        
    def draw_vision_cone(screen, guard_pos, fov_angle, length, cell_size):
        angle_rad = np.deg2rad(fov_angle/2)
        directions = [
            (np.cos(angle_rad)), 
            (np.cos(-angle_rad))
        ]
        # Calculate cone endpoints and draw