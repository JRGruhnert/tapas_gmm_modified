from enum import Enum
from functools import cached_property
import json
import pathlib
from loguru import logger
import torch
from tapas_gmm.master_project.observation import MasterObservation
from tapas_gmm.master_project.state import State
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


class Task:
    def __init__(
        self,
        name: str,
        reversed: bool,
        conditional: bool,
        overrides: list[str],
    ):
        self._name: str = name
        self._reversed: bool = reversed
        self._conditional: bool = conditional
        self._policy_name: str = "gmm"
        self._overrides: list[str] = overrides
        self._policy: GMMPolicy = self._load_policy()
        self._task_parameters: dict[str, torch.Tensor] = {}

    @property
    def name(self) -> str:
        return self._name

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
    def overrides(self) -> list[str]:
        return self._overrides

    @cached_property
    def task_parameters(self) -> dict[str, torch.Tensor]:
        if len(self._task_parameters) == 0:
            raise ValueError("Task parameters have not been initialized.")
        return self._task_parameters

    @cached_property
    def task_parameters_keys(self) -> set[str]:
        return self._task_parameters.keys()

    @classmethod
    def from_json(cls, name: str, json_data: dict) -> "Task":
        """Create a Task instance from JSON data"""
        if (
            "reversed" not in json_data
            or "conditional" not in json_data
            or "overrides" not in json_data
        ):
            raise ValueError(f"Invalid JSON data for Task {name}")
        if not isinstance(json_data["reversed"], bool):
            raise ValueError(f"Invalid JSON data for Task {name}")
        if not isinstance(json_data["conditional"], bool):
            raise ValueError(f"Invalid JSON data for Task {name}")
        if not isinstance(json_data["overrides"], list):
            raise ValueError(f"Invalid JSON data for Task {name}")
        if not all(isinstance(item, str) for item in json_data["overrides"]):
            raise ValueError(f"Invalid JSON data for Task {name}")

        return cls(
            name=name,
            reversed=json_data["reversed"],
            conditional=json_data["conditional"],
            overrides=[item for item in json_data["overrides"]],
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
            / self.name
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
        PolicyClass = import_policy(self.policy_name)
        config = self._get_config()
        policy: GMMPolicy = PolicyClass(config).to(device)

        file_name = self._policy_checkpoint_name()  # type: ignore
        logger.info("Loading policy checkpoint from {}", file_name)
        policy.from_disk(file_name)
        policy.eval()
        return policy

    def initialize_task_parameters(self, states: list[State]):
        """
        Initialize the task parameters based on the active states.
        """
        tpgmm: AutoTPGMM = self._policy.model
        # Taskparameters of the AutoTPGMM model
        tapas_tp: set[str] = set()
        for _, segment in enumerate(tpgmm.segment_frames):
            for _, frame_idx in enumerate(segment):
                print(f"Frame {frame_idx}: {tpgmm.frame_mapping[frame_idx]}")
                pos_str, rot_str = tpgmm.frame_mapping[frame_idx]
                tapas_tp.add(pos_str)
                tapas_tp.add(rot_str)
        print(f"Tapas keys: {tapas_tp}")
        for state in states:
            selected, value = state.as_tp(
                tpgmm.start_values[state.name],
                tpgmm.end_values[state.name],
                self.reversed,
                tapas_tp,
            )
            if selected:
                self._task_parameters[state.name] = value

    def distances(
        self,
        obs: MasterObservation,
        goal: MasterObservation,  # For boolean states
        states: list[State],
        pad: bool = False,
    ) -> torch.Tensor:
        task_features: list[torch.Tensor] = []
        for state in states:
            if state.name in self.task_parameters_keys:
                value = state.tp_distance(
                    obs.states[state.name],
                    self._task_parameters[state.name],
                    goal.states[state.name],
                    pad,
                )
                value = torch.tensor([value, 1.0]) if pad else torch.tensor(value)
            else:
                value = torch.tensor([0.0, 1.0]) if pad else torch.tensor(0.0)
            task_features.append(value)
        return torch.stack(task_features, dim=0)
