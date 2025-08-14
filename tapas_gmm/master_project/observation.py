import torch

from tapas_gmm.master_project.state import StateIdent
from calvin_env.envs.observation import CalvinObservation


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
