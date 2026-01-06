"""Test that observations are returned correctly in both render modes."""

import gymnasium as gym
import gym_pusht
import numpy as np

print("Testing observation image in different render modes")
print("=" * 60)

# Test 1: rgb_array mode
print("\n1. Testing render_mode='rgb_array'")
env1 = gym.make("gym_pusht/PushT2-v0", obs_type="pixels", render_mode="rgb_array")
obs1, _ = env1.reset()
print(f"   Observation type: {type(obs1)}")
print(f"   Observation shape: {obs1.shape}")
print(f"   Observation dtype: {obs1.dtype}")
assert obs1 is not None, "Observation should not be None"
assert obs1.shape == (96, 96, 3), f"Expected shape (96, 96, 3), got {obs1.shape}"
env1.close()
print("   ✓ PASS")

# Test 2: human mode
print("\n2. Testing render_mode='human'")
env2 = gym.make("gym_pusht/PushT2-v0", obs_type="pixels", render_mode="human")
obs2, _ = env2.reset()
print(f"   Observation type: {type(obs2)}")
print(f"   Observation shape: {obs2.shape}")
print(f"   Observation dtype: {obs2.dtype}")
assert obs2 is not None, "Observation should not be None"
assert obs2.shape == (96, 96, 3), f"Expected shape (96, 96, 3), got {obs2.shape}"

# Test step as well
action = env2.action_space.sample()
obs3, _, _, _, _ = env2.step(action)
print(f"   After step - Observation shape: {obs3.shape}")
assert obs3 is not None, "Observation after step should not be None"
assert obs3.shape == (96, 96, 3), f"Expected shape (96, 96, 3), got {obs3.shape}"
env2.close()
print("   ✓ PASS")

# Test 3: Verify images are in valid range
print("\n3. Testing image values")
print(f"   Min value: {obs2.min()}")
print(f"   Max value: {obs2.max()}")
assert obs2.min() >= 0, "Image values should be >= 0"
assert obs2.max() <= 255, "Image values should be <= 255"
print("   ✓ PASS")

print("\n" + "=" * 60)
print("✓ All tests passed! Observations work correctly in both modes.")
print(f"✓ Image shape is correctly (96, 96, 3) - HWC format")
