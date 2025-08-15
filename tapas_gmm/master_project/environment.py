from dataclasses import dataclass
from enum import Enum
from loguru import logger
import numpy as np
import torch

from calvin_env.envs.observation import CalvinObservation
from tapas_gmm.env.calvin import Calvin, CalvinConfig
from tapas_gmm.master_project.state import (
    State,
    StateSuccess,
    StateType,
)
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
    precise_success_threshold: float = 0.05


class MasterEnv:
    def __init__(
        self,
        config: MasterEnvConfig,
        states: list[State],
    ):
        self.config = config
        self.states = states
        self.env = Calvin(config=config.calvin_config)

        self.last_gripper_action = [1.0]  # open
        self.max_steps = 20  # TODO: Change
        self.steps_left = self.max_steps
        self.is_terminal = False

    def reset(self) -> tuple[MasterObservation, MasterObservation]:
        goal_calvin, _, _, _ = self.env.reset(settle_time=50)
        self.goal = MasterObservation(goal_calvin)

        self.current_calvin, _, _, _ = self.env.reset(settle_time=50)
        self.current = MasterObservation(self.current_calvin)
        while self.check_completion():  # Ensure that they are not the same
            self.current_calvin, _, _, _ = self.env.reset(settle_time=50)
            self.current = MasterObservation(self.current_calvin)

        self.steps_left = self.max_steps
        self.is_terminal = False
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
        try:
            prediction, _ = task.policy.predict(
                self.make_tapas_format(self.current_calvin, task, self.goal)
            )
            for action in prediction:
                if len(action.gripper) is not 0:
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
                if verbose:
                    print(self.current_calvin.ee_pose)
                    print(self.current_calvin.ee_state)
                self.current = MasterObservation(self.current_calvin)
        except FloatingPointError:
            # At some point the model crashes.
            # Have to debug if its because of bad input but seems to be not relevant for training
            print(f"Error happened!")
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
        self.steps_left -= 1
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
            if not goal_reached:
                break  # Early exit if goal is already not reached
            if state.success == StateSuccess.Area:
                if state.type is not StateType.Euler_Angle:
                    raise ValueError(
                        f"State type {state.type} doesn't support area based evaluation."
                    )
                goal_reached = self.check_area_states(
                    self.current.states[state], self.goal.states[state]
                )
            elif state.success == StateSuccess.Precise:
                # TODO: this is not correct for bool states
                goal_reached = (
                    state.distance(
                        self.current.states[state.name], self.goal.states[state.name]
                    ).item()
                    > self.config.precise_success_threshold
                )

            elif state.success == StateSuccess.Ignore:
                pass  # Probably only Quaternions cause of Model recording
                # State is not evaluated
            else:
                raise NotImplementedError(
                    f"State Success type: {state.success} is not implemented."
                )
        return goal_reached

    def check_area_states(self, x: np.ndarray, y: np.ndarray) -> bool:
        area_x = None
        area_y = None
        for name, (min_corner, max_corner) in self.env.surfaces.items():
            box_min = np.array(min_corner)
            box_max = np.array(max_corner)
            if np.all(x >= box_min) and np.all(x <= box_max):
                area_x = name
            if np.all(y >= box_min) and np.all(y <= box_max):
                area_y = name
        if area_x is None or area_y is None:
            logger.warning(
                f"Point {x} or {y} is not in any defined area. Areas: {self.env.surfaces.keys()}"
            )
        return area_x == area_y

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
        # Changing Taskparameter for reverse models
        if task is not None and goal is not None and task.reversed:
            for state_name in task.overrides:
                if state_name is "ee":
                    ee_pose = torch.cat(
                        [
                            task.task_parameters[f"{state_name}_position"],
                            task.task_parameters[f"{state_name}_rotation"],
                        ]
                    )
                    ee_state = task.task_parameters[f"{state_name}_scalar"]
                else:
                    object_poses_dict[state_name] = torch.cat(
                        [
                            goal.states[f"{state_name}_position"],
                            goal.states[f"{state_name}_rotation"],
                        ]
                    )

        object_poses = dict_to_tensordict(
            {name: torch.Tensor(pose) for name, pose in object_poses_dict.items()},
        )
        print(f"Object poses: {object_poses_dict}")

        object_states = dict_to_tensordict(
            {name: torch.Tensor([state]) for name, state in obs.object_states.items()},
        )
        print(f"Object states: {obs.object_states}")
        obs = SceneObservation(
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
        return obs
