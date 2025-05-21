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
    def render(screen, world, player_pos, guard_pos, guard_facing):
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
        
        # Draw guard & vision cone
        pygame.draw.circle(screen, StealthVisualizer.COLORS['guard'],
                          (guard_pos[1]*cell_size + cell_size//2,
                           guard_pos[0]*cell_size + cell_size//2),
                          cell_size//3)
        
        if GuardAgent.line_of_sight(guard_pos, player_pos, world.grid, guard_facing):
            StealthVisualizer.draw_vision_cone(
                screen, guard_pos, guard_facing, cell_size*8, 90, cell_size
            )
        # Draw player
        pygame.draw.circle(screen, StealthVisualizer.COLORS['player'],
                          (player_pos[1]*cell_size + cell_size//2,
                           player_pos[0]*cell_size + cell_size//2),
                          cell_size//3)
        
    @staticmethod
    def draw_vision_cone(screen, guard_pos, facing, length, fov, cell_size):
        import numpy as np
        cx = guard_pos[1]*cell_size + cell_size//2
        cy = guard_pos[0]*cell_size + cell_size//2
        base = {0:-90,1:90,2:180,3:0}[facing]
        left = np.deg2rad(base - fov/2)
        right= np.deg2rad(base + fov/2)
        pts = [
            (cx, cy),
            (cx + length*np.cos(left),  cy + length*np.sin(left)),
            (cx + length*np.cos(right), cy + length*np.sin(right)),
        ]
        s = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(s, (255,0,0,50), pts)
        screen.blit(s, (0,0))
