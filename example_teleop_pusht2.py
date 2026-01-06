import gymnasium as gym
import pygame
import gym_pusht
import time

# Create PushT2 environment with human rendering and visual goal progress feedback
env = gym.make(
    "gym_pusht/PushT2-v0",
    render_mode="human",
    visualize_goal_progress=False,
    success_threshold=0.90,
    obs_type="pixels_agent_pos",
)
observation, info = env.reset()

# Get the teleop agent
agent = env.unwrapped.teleop_agent()

print("PushT2 Teleoperation Demo")
print("=" * 50)
print("Goal 1 (Upper Left): Green → Blue when reached")
print("Goal 2 (Bottom Right): Green → Blue when reached")
print("Task: Push the T-block to cover both goals (90% coverage each)")
print("Reward: 0.5 for each goal reached (first time)")
print("Success: Both goals covered")
print("\nControls:")
print("- Move mouse close to red agent to activate teleoperation")
print("- Agent will follow your mouse cursor")
print("- Press ESC to quit")
print("=" * 50)

running = True
total_reward = 0.0

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
        print(f"observation: {observation} action: {action}")
        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Print progress when goals are reached
        if reward > 0:
            print(f"\n🎯 Reward earned: +{reward:.1f}")
            print(f"Total reward: {total_reward:.1f}/1.0")
            if info["goal_1_reached"]:
                print("✓ Goal 1 (Upper Left) REACHED!")
            if info["goal_2_reached"]:
                print("✓ Goal 2 (Bottom Right) REACHED!")

        if terminated:
            print("\n" + "=" * 50)
            print("🎉 SUCCESS! Both goals reached!")
            print(f"Final reward: {total_reward:.1f}")
            print("=" * 50)
            time.sleep(2)
            observation, info = env.reset()
            total_reward = 0.0
            print("\nEnvironment reset. Try again!")

    env.render()

    # Slow down the environment to make it easier to control
    time.sleep(0.05)

env.close()
