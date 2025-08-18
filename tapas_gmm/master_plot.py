import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse


class RolloutAnalyzer:
    def __init__(self, path: str):
        """
        Initialize analyzer with path to directory containing .pt files

        Args:
            data_path: Path to directory containing stats_epoch_*.pt files
        """
        self.data_path = path + "logs/"
        self.save_path = path
        self.batch_data = {}
        self.summary_stats = {}

    def load_all_batches(self) -> Dict[int, Dict]:
        """Load all rollout buffer files and return combined data"""
        # Updated pattern for new file format
        pattern = os.path.join(self.data_path, "stats_epoch_*.pt")
        files = glob.glob(pattern)

        if not files:
            print(f"No rollout buffer files found in {self.data_path}")
            print(f"Looking for pattern: stats_epoch_*.pt")
            # List all files in directory for debugging
            if os.path.exists(self.data_path):
                all_files = os.listdir(self.data_path)
                print(f"Available files: {all_files}")
            return {}

        print(f"Found {len(files)} rollout files")

        for file_path in sorted(files):
            try:
                # Extract epoch number from filename
                filename = os.path.basename(file_path)
                epoch = int(filename.split("_")[-1].split(".")[0])

                # Load the .pt file
                data = torch.load(file_path, map_location="cpu")

                # Convert tensors to numpy for easier processing
                batch_data = {
                    "actions": data["actions"].numpy(),
                    "logprobs": data["logprobs"].numpy(),
                    "values": data["values"].numpy(),
                    "rewards": data["rewards"].numpy(),
                    "terminals": data["terminals"].numpy(),
                }

                self.batch_data[epoch] = batch_data
                print(f"Loaded epoch {epoch}: {len(batch_data['rewards'])} timesteps")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        return self.batch_data

    def compute_batch_stats(self, batch_data: Dict) -> Dict:
        """Compute statistics for a single batch"""
        rewards = batch_data["rewards"]
        terminals = batch_data["terminals"]
        values = batch_data["values"]
        actions = batch_data["actions"]

        # Calculate episode statistics
        episode_rewards = []
        episode_lengths = []
        episode_successes = []

        current_episode_reward = 0
        current_episode_length = 0

        for i, (reward, terminal) in enumerate(zip(rewards, terminals)):
            current_episode_reward += reward
            current_episode_length += 1

            if terminal:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                # TODO: Remove hardcoded 50 as success threshold
                episode_successes.append(1 if reward >= 50.0 else 0)
                current_episode_reward = 0
                current_episode_length = 0

        return {
            "total_timesteps": len(rewards),
            "total_episodes": len(episode_rewards),
            "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "std_episode_reward": np.std(episode_rewards) if episode_rewards else 0.0,
            "min_episode_reward": np.min(episode_rewards) if episode_rewards else 0.0,
            "max_episode_reward": np.max(episode_rewards) if episode_rewards else 0.0,
            "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "success_rate": np.mean(episode_successes) if episode_successes else 0.0,
            "mean_value": np.mean(values) if len(values) > 0 else 0.0,
            "mean_reward_per_step": np.mean(rewards) if len(rewards) > 0 else 0.0,
            "action_distribution": (
                np.bincount(actions.argmax(axis=1))
                if len(actions.shape) > 1
                else np.bincount(actions.astype(int))
            ),
        }

    def compute_summary_stats(self) -> Dict:
        """Compute summary statistics across all batches"""
        if not self.batch_data:
            self.load_all_batches()

        # Check if we have any data
        if not self.batch_data:
            print("No batch data found. Cannot compute statistics.")
            self.summary_stats = {
                "batch_summaries": {},
                "overall": {
                    "total_timesteps": 0,
                    "total_batches": 0,
                    "total_episodes": 0,
                    "mean_episode_reward": 0.0,
                    "std_episode_reward": 0.0,
                    "min_episode_reward": 0.0,
                    "max_episode_reward": 0.0,
                    "mean_value": 0.0,
                    "std_value": 0.0,
                    "mean_success_rate": 0.0,
                    "mean_episode_length": 0.0,
                },
            }
            return self.summary_stats

        # Compute stats for each batch
        batch_summaries = {}
        all_episode_rewards = []
        all_values = []
        all_success_rates = []
        all_episode_lengths = []
        total_timesteps = 0
        total_episodes = 0

        for epoch, batch_data in self.batch_data.items():
            batch_stats = self.compute_batch_stats(batch_data)
            batch_summaries[epoch] = batch_stats

            # Collect data for overall stats
            rewards = batch_data["rewards"]
            terminals = batch_data["terminals"]
            values = batch_data["values"]

            # Extract episode rewards for this batch
            current_episode_reward = 0
            for reward, terminal in zip(rewards, terminals):
                current_episode_reward += reward
                if terminal:
                    all_episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0

            all_values.extend(values)
            all_success_rates.append(batch_stats["success_rate"])
            all_episode_lengths.append(batch_stats["mean_episode_length"])
            total_timesteps += batch_stats["total_timesteps"]
            total_episodes += batch_stats["total_episodes"]

        # Overall statistics
        overall_stats = {
            "total_timesteps": total_timesteps,
            "total_batches": len(self.batch_data),
            "total_episodes": total_episodes,
            "mean_episode_reward": (
                np.mean(all_episode_rewards) if all_episode_rewards else 0.0
            ),
            "std_episode_reward": (
                np.std(all_episode_rewards) if all_episode_rewards else 0.0
            ),
            "min_episode_reward": (
                np.min(all_episode_rewards) if all_episode_rewards else 0.0
            ),
            "max_episode_reward": (
                np.max(all_episode_rewards) if all_episode_rewards else 0.0
            ),
            "mean_value": np.mean(all_values) if all_values else 0.0,
            "std_value": np.std(all_values) if all_values else 0.0,
            "mean_success_rate": (
                np.mean(all_success_rates) if all_success_rates else 0.0
            ),
            "mean_episode_length": (
                np.mean(all_episode_lengths) if all_episode_lengths else 0.0
            ),
        }

        self.summary_stats = {
            "batch_summaries": batch_summaries,
            "overall": overall_stats,
        }

        return self.summary_stats

    def print_analysis(self):
        """Print comprehensive analysis of the rollout data"""
        if not self.summary_stats:
            self.compute_summary_stats()

        print("\n" + "=" * 50)
        print("ROLLOUT ANALYSIS SUMMARY")
        print("=" * 50)

        overall = self.summary_stats["overall"]
        print(f"Total Timesteps: {overall['total_timesteps']:,}")
        print(f"Total Batches: {overall['total_batches']}")
        print(f"Total Episodes: {overall['total_episodes']}")
        print(
            f"Mean Episode Reward: {overall['mean_episode_reward']:.2f} ± {overall['std_episode_reward']:.2f}"
        )
        print(
            f"Episode Reward Range: [{overall['min_episode_reward']:.2f}, {overall['max_episode_reward']:.2f}]"
        )
        print(f"Mean Episode Length: {overall['mean_episode_length']:.1f} steps")
        print(f"Success Rate: {overall['mean_success_rate']:.1%}")
        print(
            f"Mean Value Estimate: {overall['mean_value']:.3f} ± {overall['std_value']:.3f}"
        )

        print("\n" + "-" * 30)
        print("PER-BATCH BREAKDOWN")
        print("-" * 30)

        batch_summaries = self.summary_stats["batch_summaries"]
        for epoch in sorted(batch_summaries.keys()):
            stats = batch_summaries[epoch]
            print(
                f"Epoch {epoch:3d}: "
                f"Episodes={stats['total_episodes']:2d}, "
                f"Reward={stats['mean_episode_reward']:6.1f}, "
                f"Success={stats['success_rate']:4.1%}, "
                f"Length={stats['mean_episode_length']:4.1f}"
            )

    def plot_training_curves(self):
        """Plot training progress over time"""
        if not self.summary_stats:
            self.compute_summary_stats()

        batch_summaries = self.summary_stats["batch_summaries"]
        if not batch_summaries:
            print("No data to plot")
            return

        epochs = sorted(batch_summaries.keys())
        episode_rewards = [batch_summaries[e]["mean_episode_reward"] for e in epochs]
        success_rates = [batch_summaries[e]["success_rate"] for e in epochs]
        episode_lengths = [batch_summaries[e]["mean_episode_length"] for e in epochs]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Episode rewards
        ax1.plot(epochs, episode_rewards, "b-", alpha=0.7)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Mean Episode Reward")
        ax1.set_title("Training Progress: Episode Rewards")
        ax1.grid(True, alpha=0.3)

        # Success rate
        ax2.plot(epochs, success_rates, "g-", alpha=0.7)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Success Rate")
        ax2.set_title("Training Progress: Success Rate")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # Episode lengths
        ax3.plot(epochs, episode_lengths, "r-", alpha=0.7)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Mean Episode Length")
        ax3.set_title("Training Progress: Episode Length")
        ax3.grid(True, alpha=0.3)

        # Combined plot
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(epochs, episode_rewards, "b-", label="Episode Reward")
        line2 = ax4_twin.plot(epochs, success_rates, "g-", label="Success Rate")

        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Episode Reward", color="b")
        ax4_twin.set_ylabel("Success Rate", color="g")
        ax4.set_title("Combined Training Progress")

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc="center right")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.save_path, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Training curves saved to: {plot_path}")
        plt.show()


def entry_point():
    from tapas_gmm.utils.argparse import parse_and_build_config

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    # Build results path from config
    # results_path = f"results/{dict_config.nt.value}/{dict_config.tag}/"
    results_path = "results/gnn4/new_min_min/"
    print(f"Analyzing results from: {results_path}")

    analyzer = RolloutAnalyzer(results_path)
    analyzer.print_analysis()
    analyzer.plot_training_curves()


if __name__ == "__main__":
    entry_point()
