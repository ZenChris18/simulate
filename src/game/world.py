import numpy as np

class StealthWorld:
    def __init__(self, size=20):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self._generate_map()
        
    def _generate_map(self):
        # Generate rooms and walls
        self.grid.fill(0)
        # Add perimeter walls
        self.grid[0,:] = 2
        self.grid[-1,:] = 2
        self.grid[:,0] = 2
        self.grid[:,-1] = 2
        
        # Add random rooms
        for _ in range(4):
            x = np.random.randint(1, self.size-4)
            y = np.random.randint(1, self.size-4)
            w = np.random.randint(3, 6)
            h = np.random.randint(3, 6)
            self.grid[x:x+w, y:y+h] = 1  # Mark as room
            
    def change_layout(self):
        self._generate_map()