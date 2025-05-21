import numpy as np

def generate_world(size=10, resource_chance=0.4, hazard_chance=0.2):
    """Creates a 2D grid world with resources and hazards."""
    world = np.zeros((size, size), dtype=int)
    
    # Randomly assign tiles
    for x in range(size):
        for y in range(size):
            rand = np.random.rand()
            if rand < hazard_chance:
                world[x][y] = 2  # Hazard
            elif rand < (hazard_chance + resource_chance):
                world[x][y] = 1  # Resource
    return world