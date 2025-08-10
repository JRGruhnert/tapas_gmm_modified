"""
Extract mean start positions from every task parameter (TP) in a TP-GMM.
This script shows different ways to get the mean start positions from TP-GMM models.
"""

import numpy as np
from typing import Dict, List, Tuple
from tapas_gmm.policy.models.tpgmm import AutoTPGMM, TPGMM


def extract_mean_start_positions_from_tpgmm(
    tpgmm: AutoTPGMM,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Extract mean start positions from every task parameter (frame) in a TP-GMM.

    Parameters:
    -----------
    tpgmm : AutoTPGMM
        Trained TP-GMM model

    Returns:
    --------
    Dict[str, Dict[int, np.ndarray]]
        Dictionary with frame names as keys, and for each frame a dict of
        {component_idx: mean_position} for all Gaussian components
    """

    # Method 1: Get frame marginals (local coordinate systems)
    print("=== Method 1: Frame Marginals (Local Coordinates) ===")

    frame_marginals = tpgmm.get_frame_marginals(time_based=False)
    frame_names = (
        tpgmm._demos.frame_names
        if tpgmm._demos
        else [f"frame_{i}" for i in range(len(frame_marginals[0]))]
    )

    local_means = {}

    for segment_idx, segment_marginals in enumerate(frame_marginals):
        print(f"\nSegment {segment_idx}:")

        for frame_idx, frame_gmm in enumerate(segment_marginals):
            frame_name = (
                frame_names[frame_idx]
                if frame_idx < len(frame_names)
                else f"frame_{frame_idx}"
            )

            if frame_name not in local_means:
                local_means[frame_name] = {}

            print(f"  Frame '{frame_name}' - {len(frame_gmm.gaussians)} components:")

            for comp_idx, gaussian in enumerate(frame_gmm.gaussians):
                # Get mean in local frame coordinates
                mu, sigma = gaussian.get_mu_sigma(mu_on_tangent=False, as_np=True)

                # Store the mean (start position)
                local_means[frame_name][comp_idx] = mu

                print(
                    f"    Component {comp_idx}: μ = {mu[:6]} ..."
                )  # Show first 6 dims

    return local_means


def extract_global_mean_start_positions(
    tpgmm: AutoTPGMM, frame_transforms: np.ndarray, frame_quaternions: np.ndarray
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Extract mean start positions in global coordinates from every TP in a TP-GMM.

    Parameters:
    -----------
    tpgmm : AutoTPGMM
        Trained TP-GMM model
    frame_transforms : np.ndarray
        Frame transformations for converting to global coordinates
    frame_quaternions : np.ndarray
        Frame quaternions for converting to global coordinates

    Returns:
    --------
    Dict[str, Dict[int, np.ndarray]]
        Dictionary with frame names as keys, and for each frame a dict of
        {component_idx: global_mean_position}
    """

    print("=== Method 2: Global Coordinates ===")

    # Get marginals and transform to global coordinates
    marginals, joint_models = tpgmm.get_marginals_and_joint(
        fix_frames=True,  # Use fixed frames for start positions
        time_based=False,
        heal_time_variance=False,
        frame_trans=frame_transforms,
        frame_quats=frame_quaternions,
    )

    frame_names = (
        tpgmm._demos.frame_names
        if tpgmm._demos
        else [f"frame_{i}" for i in range(len(marginals[0]))]
    )
    global_means = {}

    for traj_idx, traj_marginals in enumerate(marginals):
        print(f"\nTrajectory {traj_idx}:")

        for frame_idx, frame_gmm in enumerate(traj_marginals):
            frame_name = (
                frame_names[frame_idx]
                if frame_idx < len(frame_names)
                else f"frame_{frame_idx}"
            )

            if frame_name not in global_means:
                global_means[frame_name] = {}

            print(f"  Frame '{frame_name}' - {len(frame_gmm.gaussians)} components:")

            for comp_idx, gaussian in enumerate(frame_gmm.gaussians):
                # Get mean in global coordinates
                mu, sigma = gaussian.get_mu_sigma(mu_on_tangent=False, as_np=True)

                global_means[frame_name][comp_idx] = mu

                print(f"    Component {comp_idx}: μ_global = {mu[:6]} ...")

    return global_means


def extract_first_timestep_means(tpgmm: AutoTPGMM) -> Dict[str, np.ndarray]:
    """
    Extract the mean positions at the first timestep for each frame.
    This gives you the "start position" for each task parameter.

    Parameters:
    -----------
    tpgmm : AutoTPGMM
        Trained TP-GMM model

    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary mapping frame names to their start position means
    """

    print("=== Method 3: First Timestep Means ===")

    # Get the overall model means
    if hasattr(tpgmm, "model") and tpgmm.model is not None:
        # Use the raw means from the model
        mu_raw = tpgmm.model.mu_raw[0]  # First component means

        frame_names = tpgmm._demos.frame_names if tpgmm._demos else []
        start_positions = {}

        # Extract per-frame start positions
        if tpgmm.config.add_time_component:
            time_start_idx = 1  # Skip time dimension
        else:
            time_start_idx = 0

        n_manifolds_per_frame = tpgmm._per_frame_manifold_n_submanis

        for frame_idx, frame_name in enumerate(frame_names):
            # Calculate the start index for this frame's data
            frame_start_idx = time_start_idx + frame_idx * n_manifolds_per_frame

            # Get position (first 3 dims of each frame)
            if frame_start_idx + 3 <= len(mu_raw):
                position_mean = mu_raw[frame_start_idx : frame_start_idx + 3]
                start_positions[frame_name] = position_mean

                print(f"Frame '{frame_name}': start position = {position_mean}")

        return start_positions

    else:
        print("No trained model found!")
        return {}


def extract_task_parameter_means_comprehensive(
    tpgmm: AutoTPGMM,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Comprehensive extraction of all types of means from task parameters.

    Returns:
    --------
    Dict[str, Dict[str, np.ndarray]]
        Nested dictionary with:
        - frame_name ->
          - 'position_mean': 3D position mean
          - 'rotation_mean': 4D quaternion mean (if rotation included)
          - 'full_mean': complete mean vector
    """

    print("=== Method 4: Comprehensive TP Means ===")

    frame_marginals = tpgmm.get_frame_marginals(time_based=False)
    frame_names = tpgmm._demos.frame_names if tpgmm._demos else []

    comprehensive_means = {}

    for frame_idx, frame_name in enumerate(frame_names):
        if frame_idx < len(frame_marginals[0]):  # Check bounds
            frame_gmm = frame_marginals[0][frame_idx]  # First segment

            # Average across all components to get overall frame mean
            all_means = []
            for gaussian in frame_gmm.gaussians:
                mu, _ = gaussian.get_mu_sigma(mu_on_tangent=False, as_np=True)
                all_means.append(mu)

            if all_means:
                avg_mean = np.mean(all_means, axis=0)

                comprehensive_means[frame_name] = {
                    "full_mean": avg_mean,
                }

                # Extract position (first 3 dimensions after time if present)
                pos_start = 1 if tpgmm.config.add_time_component else 0
                if len(avg_mean) >= pos_start + 3:
                    comprehensive_means[frame_name]["position_mean"] = avg_mean[
                        pos_start : pos_start + 3
                    ]

                # Extract rotation if not position-only
                if not tpgmm.config.position_only and len(avg_mean) >= pos_start + 7:
                    comprehensive_means[frame_name]["rotation_mean"] = avg_mean[
                        pos_start + 3 : pos_start + 7
                    ]

                print(f"Frame '{frame_name}':")
                print(
                    f"  Position mean: {comprehensive_means[frame_name].get('position_mean', 'N/A')}"
                )
                print(
                    f"  Rotation mean: {comprehensive_means[frame_name].get('rotation_mean', 'N/A')}"
                )

    return comprehensive_means


# Example usage function
def example_usage():
    """
    Example of how to use these functions with a trained TP-GMM.
    """

    # Assuming you have a trained tpgmm model
    # tpgmm = your_trained_model

    print("To extract mean start positions from your TP-GMM:")
    print()
    print("# Method 1: Local frame coordinates")
    print("local_means = extract_mean_start_positions_from_tpgmm(tpgmm)")
    print()
    print("# Method 2: Global coordinates (requires frame transforms)")
    print(
        "global_means = extract_global_mean_start_positions(tpgmm, frame_trans, frame_quats)"
    )
    print()
    print("# Method 3: First timestep means")
    print("start_positions = extract_first_timestep_means(tpgmm)")
    print()
    print("# Method 4: Comprehensive extraction")
    print("comprehensive = extract_task_parameter_means_comprehensive(tpgmm)")


if __name__ == "__main__":
    example_usage()
