from dataclasses import dataclass

import numpy as np
import torch

from tapas_gmm.env.calvin import Calvin
from tapas_gmm.master_project.state import (
    State,
    StateIdent,
    StateSpace,
)
from tapas_gmm.master_project.task import (
    TaskIdent,
    TaskSpace,
    Task,
)
from tapas_gmm.master_project.observation import MasterObservation
from tapas_gmm.master_project.evaluator import (
    StateEvaluator,
    EvaluatorConfig,
)
from tapas_gmm.master_project.sampler import SceneMaker
from tapas_gmm.policy.policy import Policy
from tapas_gmm.utils.observation import (
    CameraOrder,
    SceneObservation,
    SingleCamObservation,
    dict_to_tensordict,
    empty_batchsize,
)


@dataclass
class MasterEnvConfig:
    task_space: TaskSpace
    state_space: StateSpace
    eval_mode: bool
    pybullet_vis: bool
    debug_vis: bool
    evaluator: EvaluatorConfig
    real_time: bool


class MasterEnv:
    def __init__(
        self,
        config: MasterEnvConfig,
        states: list[State],
        tasks: list[Task],
    ):
        self.config = config
        self.env = Calvin(
            eval=config.eval_mode,
            vis=config.pybullet_vis,
            real_time=config.real_time,
        )
        self.evaluator = StateEvaluator(
            config.evaluator,
            self.env.surfaces,
            states,
            tasks,
        )
        self.scene_maker = SceneMaker(states)
        self.current: MasterObservation = None
        self.last_gripper_action = [1.0]  # open

    def reset(self) -> tuple[MasterObservation, MasterObservation]:
        sample, _, _, _ = self.env.reset(settle_time=0)
        ### Sampling goal and current state
        start_scene, goal_scene = self.scene_maker.make(sample.scene_obs)
        # Reset environment twice to get CalvinObservation (maybe find a better way)
        # NOTE do not switch order of resets!!
        calvin_goal, _, _, _ = self.env.reset(goal_scene, static=False, settle_time=50)
        calvin_curr, _, _, _ = self.env.reset(start_scene, static=False, settle_time=50)
        self.current = MasterObservation(calvin_curr)
        self.goal = MasterObservation(calvin_goal)
        self.evaluator.reset(self.current, self.goal)
        return self.current, self.goal

    def step(
        self, task: Task, verbose: bool = False
    ) -> tuple[float, bool, MasterObservation]:
        viz_dict = {}  # TODO: Make available
        task.policy.reset_episode(self.env)
        # Batch prediction for the given observation
        try:
            prediction, _ = task.policy.predict(self._to_tapas_format(task))
            for action in prediction:
                if len(action.gripper) is not 0:
                    self.last_gripper_action = action.gripper
                ee_action = np.concatenate(
                    (
                        action.ee,
                        self.last_gripper_action,
                    )
                )
                calvin_obs, _, _, _ = self.env.step(
                    ee_action, self.config.debug_vis, viz_dict
                )
                if verbose:
                    print(calvin_obs.ee_pose)
                    print(calvin_obs.ee_state)
                self.current = MasterObservation(calvin_obs)
        except FloatingPointError:
            # At some point the model crashes.
            # Have to debug if its because of bad input but seems to be not relevant for training
            print(f"Error happened!")
        reward, done = self.evaluator.evaluate(self.current)
        return reward, done, self.current

    def close(self):
        self.env.close()

    def _to_tapas_format(self, task: Task) -> SceneObservation:  # type: ignore
        """
        Convert the observation from the environment to a SceneObservation. This format is used for TAPAS.

        Returns
        -------
        SceneObservation
            The observation in common format as SceneObservation.
        """
        if self.calvin_obs.action is None:
            action = None
        else:
            action = torch.Tensor(self.calvin_obs.action)
        if self.calvin_obs.reward is None:
            reward = torch.Tensor([0.0])
        else:
            reward = torch.Tensor([self.calvin_obs.reward])

        camera_obs = {}

        for cam in self.calvin_obs._camera_names:
            self.calvin_obs._rgb[cam] = (
                self.calvin_obs._rgb[cam].transpose((2, 0, 1)) / 255
            )
            self.calvin_obs._mask[cam] = self.calvin_obs._mask[cam].astype(int)

            camera_obs[cam] = SingleCamObservation(
                **{
                    "rgb": torch.Tensor(self.calvin_obs._rgb[cam]),
                    "depth": torch.Tensor(self.calvin_obs._depth[cam]),
                    "mask": torch.Tensor(self.calvin_obs._mask[cam]).to(torch.uint8),
                    "extr": torch.Tensor(self.calvin_obs._extr[cam]),
                    "intr": torch.Tensor(self.calvin_obs._intr[cam]),
                },
                batch_size=empty_batchsize,
            )

        multicam_obs = dict_to_tensordict(
            {"_order": CameraOrder._create(self.calvin_obs._camera_names)} | camera_obs
        )

        object_pose_len = 7
        object_poses_list = self.calvin_obs._low_dim_object_poses.reshape(
            -1, object_pose_len
        )

        # TODO: Clean up this code
        # Changing Taskparameter for reverse models
        if task.reversed:
            self.calvin_obs.ee_pose = _origin_ee_tp_pose
            if task is TaskIdent.BlockDrawerBlueReversed:
                transform = self.overwrite_taskparameter(
                    self.goal.states[StateIdent.Blue_Transform]
                )
                object_poses_list[7] = torch.cat(
                    [transform, self.goal.states[StateIdent.Blue_Quat]]
                )
            elif task is TaskIdent.BlockDrawerPinkReversed:
                transform = self.overwrite_taskparameter(
                    self.goal.states[StateIdent.Pink_Transform]
                )
                object_poses_list[8] = torch.cat(
                    [transform, self.goal.states[StateIdent.Pink_Quat]]
                )
            elif task is TaskIdent.BlockDrawerRedReversed:
                transform = self.overwrite_taskparameter(
                    self.goal.states[StateIdent.Red_Transform]
                )
                object_poses_list[6] = torch.cat(
                    [transform, self.goal.states[StateIdent.Red_Quat]]
                )
            elif task is TaskIdent.BlockTableBlueReversed:
                object_poses_list[7] = torch.cat(
                    [
                        self.goal.states[StateIdent.Blue_Transform],
                        self.goal.states[StateIdent.Blue_Quat],
                    ]
                )
            elif task is TaskIdent.BlockTablePinkReversed:
                object_poses_list[8] = torch.cat(
                    [
                        self.goal.states[StateIdent.Pink_Transform],
                        self.goal.states[StateIdent.Pink_Quat],
                    ]
                )
            elif task is TaskIdent.BlockTableRedReversed:
                object_poses_list[6] = torch.cat(
                    [
                        self.goal.states[StateIdent.Red_Transform],
                        self.goal.states[StateIdent.Red_Quat],
                    ]
                )

        object_poses = dict_to_tensordict(
            {
                f"obj{i:03d}": torch.Tensor(pose)
                for i, pose in enumerate(object_poses_list)
            },
        )

        joint_pos = torch.Tensor(self.calvin_obs._joint_pos)
        joint_vel = torch.Tensor(self.calvin_obs._joint_vel)
        ee_pose = torch.Tensor(self.calvin_obs.ee_pose)
        ee_state = torch.Tensor([self.calvin_obs.ee_state])

        object_state_len = 1
        object_states_list = self.calvin_obs._low_dim_object_states.reshape(
            -1, object_state_len
        )

        object_states = dict_to_tensordict(
            {
                f"obj{i:03d}": torch.Tensor(state)
                for i, state in enumerate(object_states_list)
            },
        )

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

    def overwrite_taskparameter(self, transform: torch.Tensor) -> torch.Tensor:
        for name, (min_corner, max_corner) in self.surfaces.items():
            box_min = torch.tensor(min_corner)
            box_max = torch.tensor(max_corner)
            if torch.all(transform >= box_min) and torch.all(transform <= box_max):
                if name == "drawer_closed":
                    transform = (
                        transform.clone()  # TODO: change that. its doubled from sampler
                    )  # Create a copy to avoid modifying the original
                    transform[1] -= 0.18  # Adjust y-coordinate for closed drawer
                    return transform
        return transform
