import pygame
from simulation import run_episode
from world_generator import generate_world
from visualize import render
from agent import Explorer

# Initialize
world_size = 100  # Size of the world and agent
world = generate_world(size=world_size)
agent = Explorer(grid_size=world_size)
pygame.init()
screen = pygame.display.set_mode((400, 400))
number_of_episodes = 1000 # number of tries the agent will do to learn

# Training loop
for episode in range(number_of_episodes):
    total_reward, final_x, final_y = run_episode(world, agent, episode)
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    if total_reward < 100 and episode > 30:
        agent.epsilon = min(0.5, agent.epsilon + 0.1)

    print(f"Episode {episode}: Reward = {total_reward}")

    # Render once per episode so i can see
    render(screen, world, (final_x, final_y)) 
    pygame.display.update()

pygame.quit()
