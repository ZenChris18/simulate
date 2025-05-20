import pygame

def render(screen, world, agent_pos, cell_size=40):
    # Colors: empty (gray), resource (green), hazard (red), agent (blue)
    colors = [(100, 100, 100), (0, 200, 0), (200, 0, 0), (0, 0, 200)]
    
    for x in range(world.shape[0]):
        for y in range(world.shape[1]):
            rect = pygame.Rect(y*cell_size, x*cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, colors[world[x][y]], rect)
    
    # Draw agent (circle)
    pygame.draw.circle(screen, colors[3], 
                      (agent_pos[1]*cell_size + cell_size//2,
                       agent_pos[0]*cell_size + cell_size//2), 
                      cell_size//3)