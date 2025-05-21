import numpy as np

class StealthWorld:
    def __init__(self, size=20):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self._generate_map()
        
    def _generate_map(self):
        self.grid.fill(0)
        # Add perimeter walls
        self.grid[0,:] = 2
        self.grid[-1,:] = 2
        self.grid[:,0] = 2
        self.grid[:,-1] = 2
        
        # Create rooms with walls
        for _ in range(4):
            # Random room position with walls
            x = np.random.randint(2, self.size-6)
            y = np.random.randint(2, self.size-6)
            w = np.random.randint(3, 6)
            h = np.random.randint(3, 6)
            
            # Walls around room
            self.grid[x-1:x+w+1, y-1] = 2  # Left wall
            self.grid[x-1:x+w+1, y+h] = 2  # Right wall
            self.grid[x-1, y-1:y+h+1] = 2  # Top wall
            self.grid[x+w, y-1:y+h+1] = 2  # Bottom wall
            
            # Room floor
            self.grid[x:x+w, y:y+h] = 1
            
            # Add doorway
            door_x = x + np.random.randint(1, w-1)
            door_y = y + np.random.randint(1, h-1)
            self.grid[door_x, door_y] = 0
            
    def change_layout(self):
        self._generate_map()