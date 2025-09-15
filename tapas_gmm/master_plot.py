import os
import glob
import os
import glob
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

# Set a global style for all plots
print(plt.style.available)
plt.style.use("seaborn-v0_8")


class RolloutAnalyzer:
    def __init__(self, path: str):
        """
        Initialize analyzer with path to directory containing .pt files

        Args:
            data_path: Path to directory containing stats_epoch_*.pt files
        """
        self.data_path = path + "/logs/"
        self.save_path = path
        self.summary_stats = self.compute_summary_stats()

    def load_all_batches(self) -> dict[int, dict]:
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

        ready_data = {}
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

                ready_data[epoch] = batch_data
                print(f"Loaded epoch {epoch}: {len(batch_data['rewards'])} timesteps")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        return ready_data

    def compute_batch_stats(self, batch_data: dict) -> dict:
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

    def compute_summary_stats(self) -> dict:
        """Compute summary statistics across all batches"""
        batch_data = self.load_all_batches()

        # Check if we have any data
        if not batch_data:
            raise ValueError(
                "No batch data available for computing summary statistics."
            )
        # Compute stats for each batch
        batch_summaries = {}
        all_episode_rewards = []
        all_values = []
        all_success_rates = []
        all_episode_lengths = []
        total_timesteps = 0
        total_episodes = 0

        for epoch in sorted(batch_data.keys()):
            batch_stats = self.compute_batch_stats(batch_data[epoch])
            batch_summaries[epoch] = batch_stats

            # Collect data for overall stats
            rewards = batch_data[epoch]["rewards"]
            terminals = batch_data[epoch]["terminals"]
            values = batch_data[epoch]["values"]

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
            "total_batches": len(batch_data),
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
            "mean_sr": (np.mean(all_success_rates) if all_success_rates else 0.0),
            "mean_episode_length": (
                np.mean(all_episode_lengths) if all_episode_lengths else 0.0
            ),
            "max_sr": (np.amax(all_success_rates) if all_success_rates else 0.0),
            "sr_until_max": (
                all_success_rates[: int(np.argmax(all_success_rates)) + 1]
                if all_success_rates
                else []
            ),
            "sr_until_90": (
                all_success_rates[
                    : int(np.argmax(np.array(all_success_rates) >= 0.9)) + 1
                ]
                if any(np.array(all_success_rates) >= 0.9)
                else []
            ),
        }

        self.summary_stats = {
            "batch_summaries": batch_summaries,
            "overall": overall_stats,
        }

        return self.summary_stats

    def print_analysis(self):
        """Print comprehensive analysis of the rollout data"""
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
        print(f"Success Rate: {overall['mean_sr']:.1%}")
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


def plot_stats_vs_epoch(
    data: dict[str, dict[str, dict[str, list[float]]]],
    tag: str,
    save_path: str,
    nt: str,
):
    for name, rows in data[tag].items():
        plt.figure(figsize=(8, 5))
        x = rows["p"]
        for label, y in rows.items():
            if label != "p":
                plt.scatter(
                    x,
                    y,
                    label=label,
                )
        plt.xlabel("p")
        plt.ylabel("%")
        plt.title(f"{nt} Success Rate vs {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(save_path, f"{nt}_{tag}_sr_vs_{name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")


def plot_stats_vs_epoch_concat(
    data: dict[str, dict[str, dict[str, list[float]]]],
    ptaga: str,
    ptagb: str,
    rtagb: str,
    save_path: str,
    nt: str,
):
    plt.figure(figsize=(8, 5))
    clean_ptaga = data[find_index(data, 0.0, 0.0, ptaga)]
    clean_ptagb = data[find_index(data, 0.0, 0.0, ptagb)]
    clean_rtagb = data[find_index(data, 0.0, 0.0, rtagb)]
    plt.scatter(
        range(len(clean_ptaga["sr_until_max"])),
        clean_ptaga["sr_until_max"],
        label=ptaga,
    )
    a = plt.scatter(
        range(
            len(clean_ptaga["sr_until_max"]),
            len(clean_ptaga["sr_until_max"]) + len(clean_ptagb["sr_until_max"]),
        ),
        clean_ptagb["sr_until_max"],
        label=ptagb,
    )
    b = plt.scatter(
        range(
            len(clean_ptaga["sr_until_max"]),
            len(clean_ptaga["sr_until_max"]) + len(clean_rtagb["sr_until_max"]),
        ),
        clean_rtagb["sr_until_max"],
        label=rtagb,
    )
    # plt.axvline(
    #    x=len(clean_ptagb["sr_until_90"]) + len(clean_ptaga["sr_until_max"]),
    #    color=a.get_facecolor()[0],
    #    linestyle=":",
    #    label=f"Reach 90% ({len(clean_ptagb['sr_until_90'])})",
    # )
    # plt.axvline(
    #    x=len(clean_rtagb["sr_until_90"]) + len(clean_ptaga["sr_until_max"]),
    #    color=b.get_facecolor()[0],
    #    linestyle=":",
    #    label=f"Reach 90% ({len(clean_rtagb['sr_until_90'])})",
    # )
    plt.xlabel("Epoch")
    plt.ylabel("SR %")
    plt.title(f"{nt} Retraining from {ptaga} to {ptagb}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(save_path, f"re_{nt}_{ptaga}_to_{ptagb}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")


def plot_stats_vs_p(
    data: dict[str, dict[str, dict[str, list[float]]]],
    tag: str,
    save_path: str,
    nt: str,
):
    # Plot max success rate vs all p
    plt.figure(figsize=(8, 5))
    for name, rows in data[tag].items():
        print(rows.keys())
        x = rows["p"]
        for label, y in rows.items():
            if label != "p":
                plt.scatter(
                    x,
                    y,
                    label=f"{name}_{label}",
                )
    plt.xlabel("p")
    plt.ylabel("Success Rate")
    plt.title(f"{nt} Success Rate vs p")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(save_path, f"{nt}_{tag}_sr_vs_p.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")


def plot_stats_direct(
    data: dict[str, dict[str, dict[str, list[float]]]],
    tag: str,
    save_path: str,
):
    plt.figure(figsize=(8, 5))
    for name, rows in data[tag].items():
        x = rows["p"]
        for label, y in rows.items():
            if label != "p":
                plt.scatter(
                    x,
                    y,
                    label=label,
                )
    # Plot max success rate vs all p
    plt.figure(figsize=(8, 5))
    for name, rows in data[tag].items():
        x = rows["p"]
        for label, y in rows.items():
            if label != "p":
                plt.scatter(
                    x,
                    y,
                    label=label,
                )
    plt.xlabel("p")
    plt.ylabel("Success Rate")
    plt.title("Success Rate vs p")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(save_path, f"{tag}_sr_vs_p.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")


def plot_retrain(
    x1: list[float],
    x2: list[float],
    y1: list[float],
    y2: list[float],
    name: str,
    save_path: str,
):
    plt.figure(figsize=(8, 5))
    plt.scatter(
        x1,
        y1,
        label="P1",
    )
    plt.scatter(
        x2,
        y2,
        label="R1",
    )
    plt.xlabel("p (%)")
    plt.ylabel("Max Success Rate")
    plt.title("Max Success Rate vs p")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(save_path, f"{name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")


def find_index(loaded_stats, pe_val: float, pr_val: float, tag_val: str):
    for i, d in enumerate(loaded_stats):
        if d["pe"] == pe_val and d["pr"] == pr_val and d["tag"] == tag_val:
            return i
    return -1  # Not found


def entry_point():
    from tapas_gmm.utils.argparse import parse_and_build_config

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    save_path = f"results/{dict_config.nt.value}"
    nt = dict_config.nt.value
    p_tags = ["p1", "p2", "p3"]
    r_tags = ["r1", "r2", "r3"]
    tags = p_tags + r_tags
    # Create directories if they don't exist
    for tag in tags:
        tag_path = save_path + f"/{tag}"
        os.makedirs(tag_path, exist_ok=True)

    tags_pattern = "|".join(tags)
    pattern = re.compile(
        rf"(?P<tag>{tags_pattern})_pe_(?P<pe>[0-9.]+)_pr_(?P<pr>[0-9.]+)"
    )

    files = glob.glob(f"{save_path}/*", recursive=True)

    loaded_stats = []
    for file in files:
        match = pattern.search(file)
        if match:
            analyzer = RolloutAnalyzer(file)
            analyzer.print_analysis()
            analyzer.plot_training_curves()
            loaded_stats.append(
                {
                    **analyzer.summary_stats["overall"],
                    "pe": float(match.group("pe")),
                    "pr": float(match.group("pr")),
                    "tag": match.group("tag"),
                }
            )

    tagged_statistics = {}
    for tag in tags:
        tag_stats = [s for s in loaded_stats if s["tag"] == tag]
        tagged_statistics[tag] = {  # p1, p2, r1, r2
            "p_empty": {
                "p": [s["pe"] for s in tag_stats if s["pr"] == 0],
                "mean_sr": [s["mean_sr"] for s in tag_stats if s["pr"] == 0],
                "max_sr": [s["max_sr"] for s in tag_stats if s["pr"] == 0],
            },
            "p_rand": {
                "p": [s["pr"] for s in tag_stats if s["pe"] == 0],
                "mean_sr": [s["mean_sr"] for s in tag_stats if s["pe"] == 0],
                "max_sr": [s["max_sr"] for s in tag_stats if s["pe"] == 0],
            },
            "p_mix": {
                "p": [s["pe"] * 2 for s in tag_stats if s["pe"] == s["pr"]],
                "mean_sr": [s["mean_sr"] for s in tag_stats if s["pe"] == s["pr"]],
                "max_sr": [s["max_sr"] for s in tag_stats if s["pe"] == s["pr"]],
            },
        }
        plot_stats_vs_epoch(tagged_statistics, tag, save_path + f"/{tag}", nt)

        plot_stats_vs_p(tagged_statistics, tag, save_path + f"/{tag}", nt)

    plot_stats_vs_epoch_concat(loaded_stats, "p1", "p2", "r1", save_path + "/r1", nt)

    plot_stats_vs_epoch_concat(loaded_stats, "p2", "p1", "r2", save_path + "/r2", nt)

    plot_stats_vs_epoch_concat(loaded_stats, "p1", "p3", "r3", save_path + "/r3", nt)


if __name__ == "__main__":
    entry_point()
