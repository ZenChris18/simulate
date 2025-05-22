# utils/visualizer.py

import pygame
from ai.guard_agent import GuardAgent
import numpy as np

class StealthVisualizer:
    COLORS = {
        'floor':  (30,30,30),
        'wall':   (100,100,100),
        'room':   (60,100,60),
        'grid':   (25,25,25),
        'guard':  (200,0,0),
        'player': (0,0,200),
        'fov':    (255,0,0,50),
    }

    @staticmethod
    def render(screen, world, player_pos, guard_pos, guard_facing):
        W = screen.get_width(); N=world.size; cs=W//N
        screen.fill((0,0,0))

        for x in range(N):
            for y in range(N):
                v = world.grid[x][y]
                col = StealthVisualizer.COLORS['floor'] if v==0 \
                    else StealthVisualizer.COLORS['room'] if v==1 \
                    else StealthVisualizer.COLORS['wall']
                r = pygame.Rect(y*cs,x*cs,cs,cs)
                pygame.draw.rect(screen,col,r)
                pygame.draw.rect(screen,(0,0,0),r,1)

        # grid lines
        for i in range(N+1):
            pygame.draw.line(screen,StealthVisualizer.COLORS['grid'],(0,i*cs),(W,i*cs))
            pygame.draw.line(screen,StealthVisualizer.COLORS['grid'],(i*cs,0),(i*cs,W))

        # vision cone
        if GuardAgent.line_of_sight(guard_pos,player_pos,world.grid,guard_facing):
            StealthVisualizer.draw_vision_cone(screen,guard_pos,guard_facing,6*cs,90,cs)

        pygame.draw.circle(screen,StealthVisualizer.COLORS['guard'],
            (guard_pos[1]*cs+cs//2,guard_pos[0]*cs+cs//2),cs//3)
        pygame.draw.circle(screen,StealthVisualizer.COLORS['player'],
            (player_pos[1]*cs+cs//2,player_pos[0]*cs+cs//2),cs//3)

    @staticmethod
    def draw_vision_cone(screen, guard_pos, facing, length, fov, cs):
        cx,cy = guard_pos[1]*cs+cs//2, guard_pos[0]*cs+cs//2
        base = {0:-90,1:90,2:180,3:0}[facing]
        left, right = np.deg2rad(base-fov/2), np.deg2rad(base+fov/2)
        pts = [(cx,cy),
               (cx+length*np.cos(left),cy+length*np.sin(left)),
               (cx+length*np.cos(right),cy+length*np.sin(right))]
        s = pygame.Surface((screen.get_width(),screen.get_height()),pygame.SRCALPHA)
        pygame.draw.polygon(s,StealthVisualizer.COLORS['fov'],pts)
        screen.blit(s,(0,0))
