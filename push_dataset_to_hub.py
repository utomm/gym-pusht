"""
Push a LeRobot dataset to Hugging Face Hub.

This script loads an existing dataset from local disk and pushes it to the Hub.
"""

import argparse
import logging
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)


def push_dataset_to_hub(
    repo_id: str,
    root: Path | str = "data/datasets/",
    tags: list[str] | None = None,
    run_compute_stats: bool = False,
):
    """
    Push a local LeRobot dataset to Hugging Face Hub.

    Args:
        repo_id: Repository ID on the Hub (e.g., "username/pusht2-teleop")
        root: Root directory where the dataset is stored locally
        tags: Optional list of tags for the dataset on the Hub
        run_compute_stats: Whether to compute statistics before pushing
    """
    root = Path(root)
    dataset_path = root

    print(f"Checking for dataset at {dataset_path}...")

    # Check if dataset exists
    if not (dataset_path / "meta" / "info.json").exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Make sure the dataset has been created and saved locally first."
        )

    print("=" * 70)
    print("Push Dataset to Hugging Face Hub")
    print("=" * 70)
    print(f"Dataset path: {dataset_path}")
    print(f"Hub repository: https://huggingface.co/datasets/{repo_id}")
    print("=" * 70)

    # Load the dataset
    logging.info(f"Loading dataset from {dataset_path}")
    dataset = LeRobotDataset(repo_id=repo_id, root=root, local_files_only=True)

    print(f"\nDataset Info:")
    print(f"  Total episodes: {dataset.meta.total_episodes}")
    print(f"  Total frames: {dataset.meta.total_frames}")
    print(f"  FPS: {dataset.fps}")
    print(f"  Tasks: {list(dataset.meta.tasks.values())}")

    # Consolidate if needed (e.g., if stats need to be computed)
    if run_compute_stats:
        print("\n📊 Computing dataset statistics before pushing...")
        try:
            dataset.consolidate(run_compute_stats=True)
            print("✓ Statistics computed successfully")
        except Exception as e:
            logging.warning(f"Statistics computation failed: {e}")
            logging.warning("Continuing with push anyway...")
            try:
                dataset.consolidate(run_compute_stats=False)
            except Exception as e2:
                logging.warning(f"Consolidation without stats also failed: {e2}")

    # Push to hub
    print(f"\n📤 Pushing dataset to Hub: {repo_id}")
    print("This may take a while depending on dataset size...")

    try:
        dataset.push_to_hub(tags=tags)
        print("\n" + "=" * 70)
        print("✅ Dataset successfully pushed to Hub!")
        print(f"🔗 View at: https://huggingface.co/datasets/{repo_id}")
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ Failed to push dataset: {e}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Push a LeRobot dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repository ID on the Hub (e.g., 'username/pusht2-teleop')",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/datasets/",
        help="Root directory where the dataset is stored locally",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=None,
        help="Optional tags for the dataset (e.g., 'pusht' 'teleoperation')",
    )
    parser.add_argument(
        "--compute-stats",
        action="store_true",
        help="Compute dataset statistics before pushing",
    )

    args = parser.parse_args()

    push_dataset_to_hub(
        repo_id=args.repo_id,
        root=args.root,
        tags=args.tags,
        run_compute_stats=args.compute_stats,
    )
