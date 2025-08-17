from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from loguru import logger
import numpy as np
import re
import torch

from calvin_env.envs.observation import CalvinObservation
from build.lib.tapas_gmm.master_project import task
from tapas_gmm.env.calvin import Calvin, CalvinConfig
from tapas_gmm.master_project.state import State
from tapas_gmm.master_project.task import Task
from tapas_gmm.master_project.observation import MasterObservation
from tapas_gmm.utils.observation import (
    CameraOrder,
    SceneObservation,
    SingleCamObservation,
    dict_to_tensordict,
    empty_batchsize,
)


class RewardMode(Enum):
    SPARSE = 0
    RANGE = 1
    ONOFF = 2


@dataclass
class MasterEnvConfig:
    calvin_config: CalvinConfig
    debug_vis: bool
    # Reward Settings
    reward_mode: RewardMode = RewardMode.SPARSE
    max_reward: float = 100.0
    min_reward: float = 0.0


class MasterEnv:
    def __init__(
        self,
        config: MasterEnvConfig,
        states: list[State],
        tasks: list[Task],
    ):
        self.config = config
        self.states = states
        self.tasks = tasks
        self.env = Calvin(config=config.calvin_config)

        self.last_gripper_action = [1.0]  # open
        self.steps_left = self.max_steps
        self.terminal = False

    @cached_property
    def max_steps(self) -> int:
        """Calculate the maximum number of steps allowed in the environment. Its a quick estimated based on the assumption that every tapas task is half a real task."""
        max_steps = 0
        for task in self.tasks:
            if task.conditional:
                max_steps += 3
            else:
                max_steps += 1
        return max_steps

    def reset(self) -> tuple[MasterObservation, MasterObservation]:
        goal_calvin, _, _, _ = self.env.reset(settle_time=50)
        self.goal = MasterObservation(goal_calvin)

        self.current_calvin, _, _, _ = self.env.reset(settle_time=50)
        self.current = MasterObservation(self.current_calvin)
        while self.check_completion():  # Ensure that they are not the same
            self.current_calvin, _, _, _ = self.env.reset(settle_time=50)
            self.current = MasterObservation(self.current_calvin)

        self.steps_left = self.max_steps
        self.terminal = False
        return self.current, self.goal

    def wrapped_reset(self) -> SceneObservation:  # type: ignore
        """Resets the environment for data collection"""
        self.reset()
        return self.make_tapas_format(self.current_calvin)

    def step(
        self, task: Task, verbose: bool = False
    ) -> tuple[float, bool, MasterObservation]:
        viz_dict = {}  # TODO: Make available
        task.policy.reset_episode(self.env)
        # Batch prediction for the given observation
        obs_normal = self.make_tapas_format(self.current_calvin, task, self.goal)
        obs_override = self.make_tapas_format(self.current_calvin)

        # Compare EVERYTHING
        print("=== NUCLEAR COMPARISON ===")
        print(
            f"EE Pose equal: {torch.allclose(obs_normal.ee_pose, obs_override.ee_pose)}"
        )
        print(
            f"EE State equal: {torch.allclose(obs_normal.gripper_state, obs_override.gripper_state)}"
        )

        # Check object poses
        for key in obs_normal.object_poses.keys():
            equal = torch.allclose(
                obs_normal.object_poses[key], obs_override.object_poses[key]
            )
            print(f"Object {key} equal: {equal}")
            if not equal:
                print(f"  Normal: {obs_normal.object_poses[key]}")
                print(f"  Override: {obs_override.object_poses[key]}")

        # Test predictions on both
        pred_normal, _ = task.policy.predict(obs_normal)
        pred_override, _ = task.policy.predict(obs_override)

        print(f"Predictions equal: {torch.allclose(pred_normal, pred_override)}")
        print(f"Normal prediction: {pred_normal[:3]}")
        print(f"Override prediction: {pred_override[:3]}")

        try:
            prediction, _ = task.policy.predict(
                self.make_tapas_format(self.current_calvin, task, self.goal)
            )
            for action in prediction:
                if len(action.gripper) != 0:
                    self.last_gripper_action = action.gripper
                ee_action = np.concatenate(
                    (
                        action.ee,
                        self.last_gripper_action,
                    )
                )
                self.current_calvin, _, _, _ = self.env.step(
                    ee_action, self.config.debug_vis, viz_dict
                )
                self.current = MasterObservation(self.current_calvin)

        except FloatingPointError:
            # At some point the model crashes.
            # Have to debug if its because of bad input but seems to be not relevant for training
            print(f"Error happened!")
        self.steps_left -= 1
        reward, done = self.evaluate()
        return reward, done, self.current

    def wrapped_direct_step(
        self, action: np.ndarray, verbose: bool = False
    ) -> SceneObservation:  # type: ignore
        """Directly step the environment with an action."""
        logger.warning(
            "Direct stepping the environment can break internal states and is just for tapas purposes."
        )
        next_calvin, step_reward, _, _ = self.env.step(action, self.config.debug_vis)
        ee_delta = self.env.compute_ee_delta(self.current_calvin, next_calvin)
        self.current_calvin.action = torch.Tensor(ee_delta)
        self.current_calvin.reward = torch.Tensor([step_reward])
        if verbose:
            print(self.current_calvin.ee_pose)
            print(self.current_calvin.ee_state)
        tapas_obs = self.make_tapas_format(self.current_calvin)
        self.current_calvin = next_calvin
        self.current = MasterObservation(self.current_calvin)
        return tapas_obs

    def close(self):
        self.env.close()

    def evaluate(self) -> tuple[float, bool]:
        if self.terminal:
            raise UserWarning(
                "Episode already ended. Please reset the evaluator with the new goal and state."
            )
        if self.config.reward_mode is RewardMode.SPARSE:
            completion = self.check_completion()
            if completion:  # Success
                self.terminal = completion
                # print(f"success {self.steps_left}")
                return self.config.max_reward, self.terminal
            else:
                if self.steps_left == 0:  # Failure
                    self.terminal = True
                    # print(f"failure {self.steps_left}")
                    return self.config.min_reward, self.terminal
                else:  # Normal Step
                    self.terminal = False
                    # print(f"normal {self.steps_left}")
                    return self.config.min_reward, self.terminal
        if self.config.reward_mode is RewardMode.ONOFF:
            raise NotImplementedError("Reward Mode not implemented.")
        if self.config.reward_mode is RewardMode.RANGE:
            raise NotImplementedError("Reward Mode not implemented.")

    def check_completion(
        self,
    ) -> bool:
        ##### Checking if goal is reached
        goal_reached = True
        for state in self.states:
            goal_reached = state.evaluate_success_condition(
                self.current.states[state.name],
                self.goal.states[state.name],
                self.env.surfaces,
            )
            if not goal_reached:
                break  # Early exit if goal is already not reached
        return goal_reached

    def make_tapas_format(self, obs: CalvinObservation, task: Task = None, goal: MasterObservation = None) -> SceneObservation:  # type: ignore
        """
        Convert the observation from the environment to a SceneObservation. This format is used for TAPAS.

        Returns
        -------
        SceneObservation
            The observation in common format as SceneObservation.
        """
        if obs.action is None:
            action = None
        else:
            action = torch.Tensor(obs.action)

        if obs.reward is None:
            reward = torch.Tensor([0.0])
        else:
            reward = torch.Tensor([obs.reward])
        joint_pos = torch.Tensor(obs.joint_pos)
        joint_vel = torch.Tensor(obs.joint_vel)
        ee_pose = torch.Tensor(obs.ee_pose)
        ee_state = torch.Tensor([obs.ee_state])
        camera_obs = {}
        for cam in obs.camera_names:
            rgb = obs.rgb[cam].transpose((2, 0, 1)) / 255
            mask = obs.mask[cam].astype(int)

            camera_obs[cam] = SingleCamObservation(
                **{
                    "rgb": torch.Tensor(rgb),
                    "depth": torch.Tensor(obs.depth[cam]),
                    "mask": torch.Tensor(mask).to(torch.uint8),
                    "extr": torch.Tensor(obs.extr[cam]),
                    "intr": torch.Tensor(obs.intr[cam]),
                },
                batch_size=empty_batchsize,
            )

        multicam_obs = dict_to_tensordict(
            {"_order": CameraOrder._create(obs.camera_names)} | camera_obs
        )
        object_poses_dict = obs.object_poses
        object_states_dict = obs.object_states
        if task is not None and goal is not None and task.reversed:
            # NOTE: This is only a hack to make reversed tapas models work
            # TODO: Update this when possible
            logger.debug(f"Overriding Tapas Task {task.name}")
            for state_name, state_value in task.overrides.items():
                match_position = re.search(r"(.+?)_(?:position)", state_name)
                match_rotation = re.search(r"(.+?)_(?:rotation)", state_name)
                match_scalar = re.search(r"(.+?)_(?:scalar)", state_name)
                if state_name == "ee_position":
                    ee_pose = torch.cat(
                        [
                            torch.Tensor(state_value),
                            ee_pose[3:],
                        ]
                    )
                elif state_name == "ee_rotation":
                    ee_pose = torch.cat(
                        [
                            ee_pose[:3],
                            torch.Tensor(state_value),
                        ]
                    )
                elif state_name == "ee_scalar":
                    ee_state = torch.Tensor(state_value)

                # TODO: Evaluate if goal state is correct here
                elif match_position:
                    object_poses_dict[match_position.group(1)] = np.concatenate(
                        [
                            goal.states[f"{match_position.group(1)}_position"].numpy(),
                            object_poses_dict[match_position.group(1)][3:],
                        ]
                    )
                elif match_rotation:
                    object_poses_dict[match_rotation.group(1)] = np.concatenate(
                        [
                            object_poses_dict[match_rotation.group(1)][:3],
                            goal.states[f"{match_rotation.group(1)}_rotation"].numpy(),
                        ]
                    )
                elif match_scalar:
                    object_states_dict[match_scalar.group(1)] = goal.states[
                        f"{match_scalar.group(1)}_scalar"
                    ].numpy()
                else:
                    raise ValueError(f"Unknown state name: {state_name}")

        object_poses = dict_to_tensordict(
            {
                name: torch.Tensor(pose)
                for name, pose in sorted(object_poses_dict.items())
            },
        )
        object_states = dict_to_tensordict(
            {
                name: torch.Tensor([state])
                for name, state in sorted(object_states_dict.items())
            },
        )

        return SceneObservation(
            feedback=reward,
            action=action,
            cameras=multicam_obs,
            ee_pose=ee_pose,
            gripper_state=ee_state,
            object_poses=object_poses,
            object_states=object_states,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            batch_size=empty_batchsize,
        )
