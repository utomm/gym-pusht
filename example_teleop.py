import gymnasium as gym
import pygame
import gym_pusht
import time

# Create environment with human rendering
env = gym.make("gym_pusht/PushT-v0", render_mode="human")
observation, info = env.reset()

# Get the teleop agent
agent = env.unwrapped.teleop_agent()

running = True
while running:
    # Handle pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Get action from teleop agent (mouse position)
    action = agent.act(observation)

    # Only step the environment if teleoperation is active
    if action is not None:
        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.render()

    # Slow down the environment to make it easier to control
    time.sleep(0.05)

env.close()
