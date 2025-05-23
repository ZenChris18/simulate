import pygame
from game.simulation import StealthSimulation
from utils.visualizer import StealthVisualizer

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    simulation = StealthSimulation(world_size=32)
    clock = pygame.time.Clock() 
    
    running = True
    while running:
        clock.tick(15)
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Run single iteration
        simulation.hybrid_loop()
        
        # Render
        StealthVisualizer.render(screen, 
                               simulation.world,
                               simulation.player_pos,
                               simulation.guard_pos,
                               simulation.guard.facing)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()