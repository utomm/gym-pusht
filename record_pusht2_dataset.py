"""
Record PushT2 episodes using teleoperation for LeRobot dataset.

This script allows you to create a dataset by controlling the agent with your mouse.
Episodes are saved in LeRobot format and can be used for training policies.
"""

import time
import logging
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import pygame
import torch

import gym_pusht
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)


def record_pusht2_teleop(
    repo_id: str,
    root: Path | str = "datasets/",
    num_episodes: int = 50,
    fps: int = 10,
    episode_time_s: float = 60,
    obs_type: str = "pixels",
    render_mode: str = "rgb_array",
    video: bool = True,
    push_to_hub: bool = False,
    run_compute_stats: bool = False,  # Disabled by default due to LeRobot compatibility
    visualize_goal_progress: bool = True,
    success_threshold: float = 0.90,
    resume: bool = False,  # Whether to resume recording from existing dataset
):
    """
    Record PushT2 episodes using teleoperation.

    Args:
        repo_id: Repository ID for the dataset (e.g., "username/pusht2-teleop")
        root: Root directory to save the dataset
        num_episodes: Number of episodes to record
        fps: Frames per second for recording
        episode_time_s: Maximum time for each episode in seconds
        obs_type: Observation type ("pixels", "state", or "environment_state_agent_pos")
        render_mode: Rendering mode ("rgb_array" or "human")
        video: Whether to save videos
        push_to_hub: Whether to push dataset to Hugging Face Hub
        run_compute_stats: Whether to compute dataset statistics
        visualize_goal_progress: Show visual feedback when goals are reached
        success_threshold: Coverage threshold for goals
        resume: If True, resume recording from existing dataset. If False, create new dataset.
    """
    root = Path(root)
    # root.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = gym.make(
        "gym_pusht/PushT2-v0",
        obs_type=obs_type,
        render_mode=render_mode,
        visualize_goal_progress=visualize_goal_progress,
        success_threshold=success_threshold,
    )

    # Get teleop agent
    teleop_agent = env.unwrapped.teleop_agent()

    # Define features for the dataset
    # Note: episode_index, frame_index, index, and task_index are auto-managed by LeRobotDataset
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),  # [agent_x, agent_y]
            "names": {"motors": ["motor_0", "motor_1"]},
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),  # [target_x, target_y]
            "names": {"motors": ["motor_0", "motor_1"]},
        },
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    }

    # Add image features if using pixels
    if obs_type in ["pixels", "pixels_agent_pos"]:
        image_shape = (96, 96, 3)  # height, width, channels (HWC format for video)
        features["observation.image"] = {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }

    # Create or load dataset
    root = Path(root)
    dataset_path = root
    # Check if dataset exists by looking for the metadata file
    dataset_exists = (dataset_path / "meta" / "info.json").exists()

    if resume and dataset_exists:
        logging.info(f"Resuming dataset: {repo_id}")
        print(f"Loading existing dataset from: {dataset_path}")
        dataset = LeRobotDataset(repo_id=repo_id, root=root, local_files_only=True)
        # Start image writer for new episodes
        dataset.start_image_writer(num_processes=0, num_threads=4)
        # Create episode buffer for new episodes
        dataset.episode_buffer = dataset.create_episode_buffer()
        start_episode = dataset.meta.total_episodes
        print(f"Existing episodes: {start_episode}")
        print(f"Will record {num_episodes} more episodes")
    else:
        if resume and not dataset_exists:
            logging.warning(
                f"Dataset not found at {dataset_path}, creating new dataset instead"
            )
        logging.info(f"Creating new dataset: {repo_id}")
        print(f"Dataset will be saved to: {dataset_path}")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            features=features,
            use_videos=video,
        )
        start_episode = 0

    print("\n" + "=" * 70)
    print("PushT2 Teleoperation Recording")
    print("=" * 70)
    print(f"Dataset: {repo_id}")
    print(f"Starting from episode: {start_episode}")
    print(f"Episodes to record: {num_episodes}")
    print(f"FPS: {fps}")
    print(f"Observation type: {obs_type}")
    print("\nControls:")
    print("  - Move mouse close to the red agent to start")
    print("  - Agent will follow your mouse")
    print("  - Complete episode by reaching both goals")
    print("  - Press ESC to skip episode or 'q' to quit")
    print("=" * 70)

    # If using human render mode, create window
    if render_mode == "human":
        pygame.init()
        pygame.display.set_caption("PushT2 Recording")

    recorded_episodes = 0

    try:
        while recorded_episodes < num_episodes:
            print(f"\n📹 Recording Episode {recorded_episodes + 1}/{num_episodes}")
            print("-" * 70)

            # Reset environment
            seed = np.random.randint(0, 100000)
            observation, info = env.reset(seed=seed)

            episode_frames = []
            frame_index = 0  # Track frame index for consistent timestamps
            timestamp = 0.0
            start_episode_t = time.perf_counter()
            skip_episode = False
            quit_recording = False

            while timestamp < episode_time_s:
                start_loop_t = time.perf_counter()

                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_recording = True
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            skip_episode = True
                            print("⏭️  Skipping episode...")
                            break
                        if event.key == pygame.K_q:
                            quit_recording = True
                            break

                if quit_recording or skip_episode:
                    break

                # Get action from teleop agent
                action = teleop_agent.act(observation)

                # Only step if teleoperation is active
                if action is not None:
                    # Convert action to numpy array
                    action_array = np.array(action, dtype=np.float32)

                    # Step environment
                    next_observation, reward, terminated, truncated, info = env.step(
                        action_array
                    )

                    # Extract state observation (agent position only)
                    state_obs = info["pos_agent"]  # [agent_x, agent_y]

                    # Use frame-based timestamp for consistency (frame_index / fps)
                    # This ensures timestamps are exactly 1/fps apart, avoiding floating-point drift
                    frame_timestamp = frame_index / fps

                    # Create frame
                    frame = {
                        "observation.state": torch.from_numpy(
                            np.array(state_obs, dtype=np.float32)
                        ),
                        "action": torch.from_numpy(action_array),
                        "next.reward": torch.tensor([reward], dtype=torch.float32),
                        "next.done": torch.tensor(
                            [terminated or truncated], dtype=torch.bool
                        ),
                        "next.success": torch.tensor(
                            [info.get("is_success", False)], dtype=torch.bool
                        ),
                        "timestamp": torch.tensor(
                            [frame_timestamp], dtype=torch.float32
                        ),
                    }

                    # Add image if using pixels
                    if obs_type in ["pixels", "pixels_agent_pos"]:
                        if obs_type == "pixels":
                            img = next_observation  # Use next_observation after step
                        else:
                            img = next_observation["pixels"]
                        # Keep in HWC format (height, width, channels) for video
                        frame["observation.image"] = torch.from_numpy(img)

                    episode_frames.append(frame)
                    frame_index += 1  # Increment frame counter

                    # Show progress
                    if reward > 0:
                        print(
                            f"  🎯 Reward: +{reward:.1f} | Goals: {info['goal_1_reached']}, {info['goal_2_reached']}"
                        )

                    observation = next_observation

                    # Check if episode is done
                    if terminated:
                        print(
                            f"  ✅ Episode complete! Success: {info.get('is_success', False)}"
                        )
                        break

                # Render
                if render_mode == "human":
                    env.render()

                # Control frame rate
                dt_s = time.perf_counter() - start_loop_t
                sleep_time = (1.0 / fps) - dt_s
                if sleep_time > 0:
                    time.sleep(sleep_time)

                timestamp = time.perf_counter() - start_episode_t

            if quit_recording:
                print("\n⏹️  Recording stopped by user")
                break

            if skip_episode:
                print("Episode skipped")
                continue

            # Save episode if we have frames
            if len(episode_frames) > 0:
                print(f"  💾 Saving episode with {len(episode_frames)} frames...")

                # Add frames to dataset
                for frame in episode_frames:
                    dataset.add_frame(frame)

                # Save episode
                dataset.save_episode(task="PushT2")
                recorded_episodes += 1
                print(f"  ✓ Episode {recorded_episodes} saved")

                # Wait before next episode
                if recorded_episodes < num_episodes:
                    print("\n  Waiting 2 seconds before next episode...")
                    time.sleep(2)
            else:
                print("  ⚠️  No frames recorded (teleoperation not activated)")

    finally:
        env.close()
        if render_mode == "human":
            pygame.quit()

    # Consolidate dataset
    print("\n" + "=" * 70)
    print("📊 Finalizing dataset...")
    # Note: run_compute_stats can cause issues with timestamp checking in some LeRobot versions
    # Skipping stats computation for now - stats can be computed later if needed
    try:
        if run_compute_stats:
            logging.info("Computing dataset statistics")
        dataset.consolidate(run_compute_stats=run_compute_stats)
    except TypeError as e:
        if "stack()" in str(e):
            logging.warning(f"Stats computation failed (known issue): {e}")
            logging.warning("Re-consolidating without computing stats...")
            dataset.consolidate(run_compute_stats=False)
        else:
            raise

    print(f"✓ Dataset saved to: {dataset.root}")
    print(f"✓ Total episodes: {dataset.num_episodes}")
    print(f"✓ Total frames: {dataset.num_frames}")

    # Push to hub if requested
    if push_to_hub:
        print("\n📤 Pushing to Hugging Face Hub...")
        dataset.push_to_hub()
        print(f"✓ Dataset pushed to: https://huggingface.co/datasets/{repo_id}")

    print("=" * 70)
    print("🎉 Recording complete!")
    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Record PushT2 dataset with teleoperation"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repository ID (e.g., 'username/pusht2-teleop')",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/datasets/",
        help="Root directory for dataset storage",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second",
    )
    parser.add_argument(
        "--episode-time",
        type=float,
        default=60.0,
        help="Maximum episode time in seconds",
    )
    parser.add_argument(
        "--obs-type",
        type=str,
        default="pixels",
        choices=["pixels", "state", "environment_state_agent_pos", "pixels_agent_pos"],
        help="Observation type",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
        help="Render mode",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video recording",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--visualize-progress",
        action="store_true",
        help="Disable visual feedback for goal progress",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.90,
        help="Coverage threshold for goals",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume recording from existing dataset instead of creating new one",
    )

    args = parser.parse_args()

    record_pusht2_teleop(
        repo_id=args.repo_id,
        root=args.root,
        num_episodes=args.num_episodes,
        fps=args.fps,
        episode_time_s=args.episode_time,
        obs_type=args.obs_type,
        render_mode=args.render_mode,
        video=not args.no_video,
        push_to_hub=args.push_to_hub,
        visualize_goal_progress=args.visualize_progress,
        success_threshold=args.success_threshold,
        resume=args.resume,
    )
