from world_generator import generate_world
import numpy as np

def run_episode(world, agent, max_steps=100, episode=0):
    size = world.shape[0]
    x, y = np.random.randint(0, size, 2)  # Random start
    total_reward = 0
    
    for _ in range(max_steps):
        # Get current state and action
        state = (x, y)
        action = agent.get_action(state)
        
        # Calculate new position
        new_x, new_y = x, y

        if action == 0: 
            new_x = max(0, x-1)    # Up
        elif action == 1: 
            new_x = min(size-1, x+1)  # Down
        elif action == 2: 
            new_y = max(0, y-1)  # Left
        else: 
            new_y = min(size-1, y+1)  # Right
        
        # Calculate reward
        tile_type = world[new_x][new_y]
        if tile_type == 1:
            world[new_x][new_y] = 0  # Collect resource
            reward = 20  # Resource
            for _ in range(np.random.randint(1,4)):
                rx, ry = np.random.randint(0, size, 2)
                if world[rx][ry] == 0:  # Only spawn in empty cells
                    world[rx][ry] = 1

            if np.sum(world == 1) < 15:
                for _ in range(8):
                    rx, ry = np.random.randint(0, size, 2)
                    if world[rx][ry] == 0:
                        world[rx][ry] = 1

        elif tile_type == 2:
            reward = -15 
            # Add hazard neighborhood penalty
            hazard_decay = max(0.1, 1 - (episode / 30))
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    if (dx != 0 or dy != 0):  # Exclude current cell
                        if (0 <= new_x+dx < size) and (0 <= new_y+dy < size):
                            if world[new_x+dx][new_y+dy] == 2:
                                reward -= 2 * hazard_decay
        else:
            reward = -0.1 # for movement
            if world[new_x][new_y] == 1:
                reward = -0.5
            world[new_x][new_y] = 1 # this means the tile is visited, now marked as visited

        # Update Q-table
        agent.update(state, action, reward, (new_x, new_y))
        
        # Update position and total reward
        x, y = new_x, new_y
        total_reward += reward
    
    return total_reward, x, y