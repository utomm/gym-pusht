import gymnasium as gym
import gym_pusht

# Test the modified PushT2 environment
env = gym.make("gym_pusht/PushT2-v0", render_mode="rgb_array")
observation, info = env.reset()

print("Environment initialized successfully!")
print(f"Goal 1 position: {info['goal_pose_1'][:2]}")
print(f"Goal 2 position: {info['goal_pose_2'][:2]}")
print(f"Initial state - Goal 1 reached: {info.get('goal_1_reached', False)}")
print(f"Initial state - Goal 2 reached: {info.get('goal_2_reached', False)}")

# Test a few steps
for i in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if reward > 0:
        print(f"Step {i}: Reward = {reward:.2f}")
        print(f"  Coverage Goal 1: {info['coverage_goal_1']:.2f}")
        print(f"  Coverage Goal 2: {info['coverage_goal_2']:.2f}")
        print(f"  Goal 1 reached: {info['goal_1_reached']}")
        print(f"  Goal 2 reached: {info['goal_2_reached']}")

    if terminated:
        print(f"Success! Both goals reached at step {i}")
        break

env.close()
print("\nTest completed successfully!")
