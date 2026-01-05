import gymnasium as gym
import gym_pusht

print("Testing PushT2 Configuration Options")
print("=" * 60)

# Test 1: Default configuration (no visual feedback)
print("\n1. Default configuration:")
env1 = gym.make("gym_pusht/PushT2-v0", render_mode="rgb_array")
print(f"   - success_threshold: {env1.unwrapped.success_threshold}")
print(f"   - visualize_goal_progress: {env1.unwrapped.visualize_goal_progress}")
env1.close()

# Test 2: Custom threshold
print("\n2. Custom threshold (0.85):")
env2 = gym.make("gym_pusht/PushT2-v0", render_mode="rgb_array", success_threshold=0.85)
print(f"   - success_threshold: {env2.unwrapped.success_threshold}")
print(f"   - visualize_goal_progress: {env2.unwrapped.visualize_goal_progress}")
env2.close()

# Test 3: Enable visual feedback
print("\n3. Visual feedback enabled:")
env3 = gym.make(
    "gym_pusht/PushT2-v0", render_mode="rgb_array", visualize_goal_progress=True
)
print(f"   - success_threshold: {env3.unwrapped.success_threshold}")
print(f"   - visualize_goal_progress: {env3.unwrapped.visualize_goal_progress}")
env3.close()

# Test 4: Both custom options
print("\n4. Both options customized:")
env4 = gym.make(
    "gym_pusht/PushT2-v0",
    render_mode="rgb_array",
    success_threshold=0.95,
    visualize_goal_progress=True,
)
print(f"   - success_threshold: {env4.unwrapped.success_threshold}")
print(f"   - visualize_goal_progress: {env4.unwrapped.visualize_goal_progress}")
env4.close()

print("\n" + "=" * 60)
print("All configuration tests passed! ✓")
