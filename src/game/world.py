# game/world.py

import numpy as np

class StealthWorld:
    def __init__(self, size=20, room_count=4):
        self.size = size
        self.room_count = room_count
        self.grid = np.zeros((size,size), dtype=int)
        self._generate_map()

    def _generate_map(self):
        self.grid.fill(0)
        # perimeter
        self.grid[0,:]=self.grid[-1,:]=2
        self.grid[:,0]=self.grid[:,-1]=2

        rooms=[]; tries=0
        while len(rooms)<self.room_count and tries<100:
            tries+=1
            w,h = np.random.randint(3,6), np.random.randint(3,6)
            x,y = np.random.randint(2,self.size-w-2), np.random.randint(2,self.size-h-2)
            rect=(x-1,y-1,x+w+1,y+h+1)
            if any(not (rect[2]<r[0] or rect[0]>r[2] or rect[3]<r[1] or rect[1]>r[3]) for r in rooms):
                continue
            rooms.append(rect)
            # walls
            self.grid[x-1:x+w+1,y-1]=2
            self.grid[x-1:x+w+1,y+h]=2
            self.grid[x-1,y-1:y+h+1]=2
            self.grid[x+w,y-1:y+h+1]=2
            # floor
            self.grid[x:x+w,y:y+h]=1

            # door on a random wall
            side = np.random.choice(["top","bot","left","right"])
            if side=="top":    dx,dy = x-1, np.random.randint(y,y+h)
            elif side=="bot":  dx,dy = x+w, np.random.randint(y,y+h)
            elif side=="left": dx,dy = np.random.randint(x,x+w), y-1
            else:              dx,dy = np.random.randint(x,x+w), y+h
            self.grid[dx,dy]=0

    def change_layout(self):
        self._generate_map()
