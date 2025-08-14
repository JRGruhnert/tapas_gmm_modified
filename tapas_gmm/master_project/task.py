from enum import Enum
from functools import cached_property
import json
import pathlib
from loguru import logger
import torch
from build.lib.conf.policy.models import tpgmm
from tapas_gmm.master_project.observation import MasterObservation
from tapas_gmm.master_project.state import StateBound, StateIdent
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
    Minimal = "Minimal"
    All = "All"
    Unused = "Unused"


class TaskIdent(Enum):
    CloseDrawer = "CloseDrawer"
    OpenDrawer = "OpenDrawer"
    BackFromCloseDrawer = "BackFromCloseDrawer"
    BackFromOpenDrawer = "BackFromOpenDrawer"
    PressButton = "PressButton"
    BackFromPressButton = "BackFromPressButton"
    SliderToLeft = "SliderToLeft"
    SliderToRight = "SliderToRight"
    BackFromSliderToLeft = "BackFromSliderToLeft"
    BackFromSliderToRight = "BackFromSliderToRight"
    GrabRedBlockTable = "GrabRedBlockTable"
    GrabPinkBlockTable = "GrabPinkBlockTable"
    GrabBlueBlockTable = "GrabBlueBlockTable"
    GrabRedBlockDrawer = "GrabRedBlockDrawer"
    GrabPinkBlockDrawer = "GrabPinkBlockDrawer"
    GrabBlueBlockDrawer = "GrabBlueBlockDrawer"
    PlaceRedBlockTable = "PlaceRedBlockTable"
    PlacePinkBlockTable = "PlacePinkBlockTable"
    PlaceBlueBlockTable = "PlaceBlueBlockTable"
    PlaceRedBlockDrawer = "PlaceRedBlockDrawer"
    PlacePinkBlockDrawer = "PlacePinkBlockDrawer"
    PlaceBlueBlockDrawer = "PlaceBlueBlockDrawer"


class Task:
    def __init__(
        self,
        ident: TaskIdent,
        reversed: bool,
        conditional: bool,
        policy_path: str,
        policy_name: str,
        overwrites: list[StateIdent],
    ):
        self._ident: TaskIdent = ident
        self._reversed: bool = reversed
        self._conditional: bool = conditional
        self._policy_path: str = policy_path
        self._policy_name: str = policy_name
        self._overwrites: list[StateIdent] = overwrites
        self._policy: GMMPolicy = self._load_policy()
        self._preconditions: dict[StateIdent, StateBound] = None

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

    @property
    def overwrites(self) -> list[StateIdent]:
        return self._overwrites

    @classmethod
    def from_json(cls, ident_value: str, json_data: dict) -> "Task":
        """Create a Task instance from JSON data"""
        if (
            "reversed" not in json_data
            or "conditional" not in json_data
            or "policy_path" not in json_data
            or "policy_name" not in json_data
            or "preconditions" not in json_data
            or "overwrites" not in json_data
        ):
            raise ValueError(f"Invalid JSON data for Task {ident_value}")
        if not isinstance(json_data["reversed"], bool):
            raise ValueError(f"Invalid JSON data for Task {ident_value}")
        if not isinstance(json_data["conditional"], bool):
            raise ValueError(f"Invalid JSON data for Task {ident_value}")
        if not isinstance(json_data["policy_path"], str):
            raise ValueError(f"Invalid JSON data for Task {ident_value}")
        if not isinstance(json_data["policy_name"], str):
            raise ValueError(f"Invalid JSON data for Task {ident_value}")
        if not isinstance(json_data["overwrites"], list):
            raise ValueError(f"Invalid JSON data for Task {ident_value}")
        if not all(isinstance(item, str) for item in json_data["overwrites"]):
            raise ValueError(f"Invalid JSON data for Task {ident_value}")

        return cls(
            ident=TaskIdent(ident_value),
            reversed=json_data["reversed"],
            conditional=json_data["conditional"],
            policy_path=json_data["policy_path"],
            policy_name=json_data["policy_name"],
            overwrites=[StateIdent(item) for item in json_data["overwrites"]],
        )

    @classmethod
    def from_json_list(cls, task_space: TaskSpace) -> list["Task"]:
        """Convert a StateSpace to a list of State objects by reading from tasks.json"""
        # Load the tasks.json file
        tasks_json_path = pathlib.Path(__file__).parent / "data" / "tasks.json"

        if not tasks_json_path.exists():
            raise FileNotFoundError(f"Tasks JSON file not found at {tasks_json_path}")

        with open(tasks_json_path, "r") as f:
            tasks_data = json.load(f)

        # Filter tasks based on the requested state space
        filtered_tasks = []

        for ident, task_data in tasks_data.items():
            # Check if this task belongs to the requested space
            task_space_str = task_data.get("space", "Unused")

            include_task = False

            if task_space == TaskSpace.Minimal and task_space_str == "Minimal":
                include_task = True
            elif task_space == TaskSpace.All and task_space_str in ["Minimal", "All"]:
                include_task = True
            elif task_space == TaskSpace.Unused:
                raise ValueError(
                    "Unused space is not supported in convert_to_tasks method"
                )

            if include_task:
                task = cls.from_json(ident, task_data)
                filtered_tasks.append(task)

        return filtered_tasks

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

    def quaternion_mean_exp_map(quaternions):
        """
        quaternions: tensor of shape [N, 4] where each row is [w, x, y, z]
        """
        # Convert quaternions to rotation vectors (axis-angle representation)
        rotation_vectors = []

        for q in quaternions:
            # Normalize quaternion
            q = q / torch.norm(q)

            # Extract angle and axis
            w, x, y, z = q
            angle = 2 * torch.acos(torch.abs(w))

            if torch.sin(angle / 2) > 1e-6:
                axis = torch.tensor([x, y, z]) / torch.sin(angle / 2)
                rotation_vector = angle * axis
            else:
                rotation_vector = torch.zeros(3)

            rotation_vectors.append(rotation_vector)

        # Average the rotation vectors
        mean_rotation_vector = torch.stack(rotation_vectors).mean(dim=0)

        # Convert back to quaternion
        angle = torch.norm(mean_rotation_vector)
        if angle > 1e-6:
            axis = mean_rotation_vector / angle
            w = torch.cos(angle / 2)
            xyz = torch.sin(angle / 2) * axis
            return torch.cat([w.unsqueeze(0), xyz])
        else:
            return torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

    @cached_property
    def state_origins(self) -> dict[StateIdent, torch.Tensor]:
        tpgmm: AutoTPGMM = self._policy.model
        if self.reversed:
            poses = tpgmm.end_object_poses
            scalars = tpgmm.end_object_scalars
        else:
            poses = tpgmm.start_object_poses
            scalars = tpgmm.start_object_scalars
        origins: dict[StateIdent, torch.Tensor] = {}
        for key, value in poses.items():
            # value shape: [N_demos, 7] where 7 = [x, y, z, qw, qx, qy, qz]
            positions = value[:, :3]
            quaternions = value[:, 3:]
            mean_position = positions.mean(dim=0)
            mean_quaternion = self.quaternion_mean_exp_map(quaternions)
            origins[StateIdent(f"{key}_euler")] = mean_position
            origins[StateIdent(f"{key}_quat")] = mean_quaternion
        for key, value in scalars.items():
            # value shape: [N_demos, 1]
            mean_scalar = value.mean(dim=0)
            origins[StateIdent(key)] = mean_scalar
        return origins

    @cached_property
    def state_task_parameters_values(self) -> dict[StateIdent, torch.Tensor]:
        # NOTE: This acts as a filter for the task parameters
        # It returns only those task parameters that are constant across all demonstrations
        # It acts as a simplistic task parameter selection
        tpgmm: AutoTPGMM = self._policy.model
        if self.reversed:
            states = tpgmm.end_object_scalars
        else:
            states = tpgmm.start_object_scalars
        states_mean = {key: value.mean(dim=0) for key, value in states.items()}
        states_max = {key: value.max(dim=0).values for key, value in states.items()}
        states_min = {key: value.min(dim=0).values for key, value in states.items()}
        states_std = {key: value.std(dim=0) for key, value in states.items()}
        state_task_parameters: dict[StateIdent, torch.Tensor] = {}

        # Define relative threshold for "constant" states (as percentage of data range)
        relative_threshold = 0.05  # 5% of the data range

        for key, value in states_mean.items():
            # Calculate data range for each dimension
            data_range = states_max[key] - states_min[key]
            # Create threshold based on data range (with minimum threshold to avoid division issues)
            threshold = relative_threshold * data_range
            # If std is small relative to data range for ALL dimensions, assume constant state
            if (states_std[key] < threshold).all():
                state_task_parameters[StateIdent(key)] = value
        return state_task_parameters

    @cached_property
    def tapas_task_parameters_values(self) -> dict[StateIdent, torch.Tensor]:
        tpgmm: AutoTPGMM = self._policy.model
        result: dict[StateIdent, torch.Tensor] = {}
        for _, segment in enumerate(tpgmm.segment_frames):
            for _, frame_idx in enumerate(segment):
                state_str = tpgmm.frame_mapping[frame_idx]
                result[StateIdent(state_str + "_euler")] = self.state_origins[
                    StateIdent(state_str + "_euler")
                ]
                result[StateIdent(state_str + "_quat")] = self.state_origins[
                    StateIdent(state_str + "_quat")
                ]
        return result

    @cached_property
    def task_parameters_values(self) -> dict[StateIdent, torch.Tensor]:
        return {
            **self.state_task_parameters_values,
            **self.tapas_task_parameters_values,
        }

    @cached_property
    def state_task_parameters(self) -> set[StateIdent]:
        return set(self.state_task_parameters_values.keys())

    @cached_property
    def tapas_task_parameters(self) -> set[StateIdent]:
        return set(self.tapas_task_parameters_values.keys())

    @cached_property
    def task_parameters(self) -> set[StateIdent]:
        return set(self.state_task_parameters + self.tapas_task_parameters)

    def distances(
        obs: MasterObservation,
        pad: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError()
        task_features: list[torch.Tensor] = []
        task_tps = self.tps[task]
        for key, converter in self.converter.items():
            if key in task_tps:
                val = converter.distance(obs.states[key], task_tps[key])
                task_value = torch.tensor([val, 0.0]) if pad else torch.tensor(val)
            else:
                val = self.distance_converter.distance(obs.states[key], obs.states[key])
                task_value = torch.tensor([val, 1.0]) if pad else torch.tensor(val)

            task_features.append(task_value)
        # Ensure consistent 2D shape: [num_states, feature_dim]
        task_features = torch.stack(task_features, dim=0)  # shape: [num_states, 2]
        features.append(task_features)
