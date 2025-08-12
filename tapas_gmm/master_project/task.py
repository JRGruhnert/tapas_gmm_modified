from enum import Enum
from functools import cached_property
import pathlib
from loguru import logger
import torch
from tapas_gmm.master_project.observation import MasterObservation
from tapas_gmm.master_project.state import StateIdent
from tapas_gmm.policy.policy import Policy
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.policy import import_policy
from tapas_gmm.policy.gmm import GMMPolicy, GMMPolicyConfig
from tapas_gmm.policy.models.tpgmm import (
    AutoTPGMM,
    AutoTPGMMConfig,
    ModelType,
    TPGMMConfig,
)


class TaskSpace(Enum):
    SMALL = 0
    ALL = 1


class TaskIdent(Enum):

    @classmethod
    def get_enum_by_index(enum_cls, index: int):
        return list(enum_cls)[index]

    CloseDrawer = 0
    OpenDrawer = 1
    BackFromCloseDrawer = 2
    BackFromOpenDrawer = 3
    PressButton = 4
    BackFromPressButton = 5
    MoveSliderToLeft = 6
    MoveSliderToRight = 7
    BackFromMoveSliderToLeft = 8
    BackFromMoveSliderToRight = 9
    GrabRedBlockTable = 10
    GrabPinkBlockTable = 11
    GrabBlueBlockTable = 12
    GrabRedBlockDrawer = 13
    GrabPinkBlockDrawer = 14
    GrabBlueBlockDrawer = 15
    PlaceRedBlockTable = 16
    PlacePinkBlockTable = 17
    PlaceBlueBlockTable = 18
    PlaceRedBlockDrawer = 19
    PlacePinkBlockDrawer = 20
    PlaceBlueBlockDrawer = 21


class Task:

    def __init__(
        self,
        ident: TaskIdent,
        reversed: bool,
        conditional: bool,
        policy_path: str,
        policy_name: str,
        preconditions: dict[StateIdent, float],
        overwrites: list[StateIdent],
    ):
        self._ident: TaskIdent = ident
        self._reversed: bool = reversed
        self._conditional: bool = conditional
        self._policy_path: str = policy_path
        self._policy_name: str = policy_name
        self._preconditions: dict[StateIdent, float] = preconditions
        self._overwrites: list[StateIdent] = overwrites
        self._policy: GMMPolicy = self._load_policy()

    @property
    def ident(self) -> TaskIdent:
        return self._ident

    @property
    def reversed(self) -> bool:
        return self._reversed

    @property
    def conditional(self) -> bool:
        return self._conditional

    @property
    def policy(self) -> Policy:
        return self._policy

    @property
    def policy_name(self) -> str:
        return self._policy_name

    @property
    def policy_path(self) -> str:
        return self._policy_path

    @property
    def preconditions(self) -> dict[StateIdent, float]:
        return self._preconditions

    @classmethod
    def convert_to_tasks(task_space: TaskSpace) -> list["Task"]:
        raise NotImplementedError(
            "This method should be implemented in subclasses to handle task conversion."
        )

    @classmethod
    def from_json(cls, json_data: dict) -> "Task":
        raise NotImplementedError(
            "This method should be implemented in subclasses to handle JSON deserialization."
        )

    def _policy_checkpoint_name(self) -> pathlib.Path:
        return (
            pathlib.Path("data")
            / self.policy_path
            / ("demos" + "_" + "gmm" + "_policy" + "-release")
        ).with_suffix(".pt")

    def _get_config(self) -> GMMPolicyConfig:
        """
        Get the configuration for the OpenDrawer policy.
        """
        return GMMPolicyConfig(
            suffix="release",
            model=AutoTPGMMConfig(
                tpgmm=TPGMMConfig(
                    n_components=20,
                    model_type=ModelType.HMM,
                    use_riemann=True,
                    add_time_component=True,
                    add_action_component=False,
                    position_only=False,
                    add_gripper_action=True,
                    reg_shrink=1e-2,
                    reg_diag=2e-4,
                    reg_diag_gripper=2e-2,
                    reg_em_finish_shrink=1e-2,
                    reg_em_finish_diag=2e-4,
                    reg_em_finish_diag_gripper=2e-2,
                    trans_cov_mask_t_pos_corr=False,
                    em_steps=1,
                    fix_first_component=True,
                    fix_last_component=True,
                    reg_init_diag=5e-4,  # 5
                    heal_time_variance=False,
                ),
            ),
            time_based=True,
            predict_dx_in_xdx_models=False,
            binary_gripper_action=True,
            binary_gripper_closed_threshold=0.95,
            dbg_prediction=False,
            # the kinematics model in RLBench is just to unreliable -> leads to mistakes
            topp_in_t_models=False,
            force_overwrite_checkpoint_config=True,  # TODO:  otherwise it doesnt work
            time_scale=1.0,
            # ---- Changing often ----
            postprocess_prediction=False,  # TODO:  abs quaternions if False else delta quaternions
            return_full_batch=True,
            batch_predict_in_t_models=True,  # Change if visualization is needed
            invert_prediction_batch=self.reversed,
        )

    def _load_policy(self) -> GMMPolicy:
        if self.policy_name is not "gmm":
            raise NotImplementedError(f"Unsupported policy type: {self.policy_name}")
        PolicyClass = import_policy(self.policy_name)
        config = self._get_config()
        policy: GMMPolicy = PolicyClass(config).to(device)

        file_name = self._policy_checkpoint_name()  # type: ignore
        logger.info("Loading policy checkpoint from {}", file_name)
        policy.from_disk(file_name)
        policy.eval()
        return policy

    @cached_property
    def state_origins(self) -> dict[StateIdent, torch.Tensor]:
        tpgmm: AutoTPGMM = self._policy.model
        eulers = tpgmm.get_frame_origins_euler(self.reversed)[0]
        quats = tpgmm.get_frame_origins_quats(self.reversed)[0]

        return None

    @cached_property
    def task_parameters_values(self) -> dict[StateIdent, torch.Tensor]:
        # TODO: add tapas taskparameter
        return self.preconditions
        result: dict[int, torch.Tensor] = {}
        for _, segment in enumerate(tpgmm.segment_frames):
            for _, frame_idx in enumerate(segment):
                # TODO: torch and check for if quaternions are needed
                transform_key, quaternion_key = State.get_tp_by_index(frame_idx)
                result[transform_key] = positions[frame_idx]
                result[quaternion_key] = quaternions[frame_idx]
        for key, value in task.value.precondition.items():
            result[key] = value
        return result

    @cached_property
    def task_parameters(self) -> list[StateIdent]:
        # TODO: add tapas taskparameter
        return self.preconditions
        result: dict[int, torch.Tensor] = {}
        for _, segment in enumerate(tpgmm.segment_frames):
            for _, frame_idx in enumerate(segment):
                # TODO: torch and check for if quaternions are needed
                transform_key, quaternion_key = State.get_tp_by_index(frame_idx)
                result[transform_key] = positions[frame_idx]
                result[quaternion_key] = quaternions[frame_idx]
        for key, value in task.value.precondition.items():
            result[key] = value
        return result

    def distances(
        obs1: MasterObservation,
        obs2: MasterObservation = None,
        pad: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError()
        task_features: list[torch.Tensor] = []
        task_tps = self.tps[task]
        for key, converter in self.converter.items():
            if key in task_tps:
                val = converter.distance(obs1.states[key], task_tps[key])
                task_value = torch.tensor([val, 0.0]) if pad else torch.tensor(val)
            else:
                val = self.distance_converter.distance(
                    obs1.states[key], obs1.states[key]
                )
                task_value = torch.tensor([val, 1.0]) if pad else torch.tensor(val)

            task_features.append(task_value)
        # Ensure consistent 2D shape: [num_states, feature_dim]
        task_features = torch.stack(task_features, dim=0)  # shape: [num_states, 2]
        features.append(task_features)
