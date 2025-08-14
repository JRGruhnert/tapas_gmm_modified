import torch

from tapas_gmm.master_project.state import StateIdent
from tapas_gmm.master_project.task import TaskIdent, Task
from calvin_env.envs.observation import CalvinObservation

from tapas_gmm.utils.observation import (
    CameraOrder,
    SceneObservation,
    SingleCamObservation,
    dict_to_tensordict,
    empty_batchsize,
)


class MasterObservation:
    __slots__ = "_states"

    def __init__(
        self,
        obs: CalvinObservation,
    ):

        self._states: dict[StateIdent, torch.Tensor] = {}
        self._states[StateIdent.EE_Euler] = torch.tensor(
            obs.ee_pose[:3], dtype=torch.float32
        )
        self._states[StateIdent.EE_Quat] = torch.tensor(
            obs.ee_pose[-4:], dtype=torch.float32
        )
        self._states[StateIdent.EE_State] = torch.tensor(
            [obs.ee_state], dtype=torch.float32
        )
        for k, pose in obs.object_poses.items():
            self._states[StateIdent(f"{k}_euler")] = torch.tensor(
                pose[:3], dtype=torch.float32
            )
            self._states[StateIdent(f"{k}_quat")] = torch.tensor(
                pose[-4:], dtype=torch.float32
            )

        for k, val in obs.object_states.items():
            self._states[StateIdent(k)] = torch.tensor([val], dtype=torch.float32)

    @property
    def states(self) -> dict[StateIdent, torch.Tensor]:
        """Returns the scalar states of the observation."""
        return self._states


def make_tapas_format(obs: CalvinObservation, task: Task = None, goal: MasterObservation = None) -> SceneObservation:  # type: ignore
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
        for state_ident in task.overwrites:
            if state_ident is StateIdent.EE_State:
                ee_pose = torch.cat(
                    [
                        task.state_origins[StateIdent.EE_Euler],
                        task.state_origins[StateIdent.EE_Quat],
                    ]
                )
                ee_state = torch.Tensor([task.state_origins[StateIdent.EE_State]])
            else:
                object_poses_dict[state_ident.value] = torch.cat(
                    [
                        goal.states[StateIdent(f"{state_ident.value}_euler")],
                        goal.states[StateIdent(f"{state_ident.value}_quat")],
                    ]
                )

    object_poses = dict_to_tensordict(
        {name: torch.Tensor(pose) for name, pose in object_poses_dict.items()},
    )

    object_states = dict_to_tensordict(
        {name: torch.Tensor(state) for name, state in obs.object_states.items()},
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
