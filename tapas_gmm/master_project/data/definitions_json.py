import json
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Union
from pathlib import Path


class RewardMode(Enum):
    SPARSE = 0
    RANGE = 1
    ONOFF = 2


class TaskSpace(Enum):
    SMALL = 0
    ALL = 1


class StateSpace(Enum):
    SMALL = 0
    ALL = 1
    UNUSED = 2


class StateType(Enum):
    Transform = "Transform"
    Quaternion = "Quaternion"
    Scalar = "Scalar"


class StateSuccess(Enum):
    AREA = "AREA"
    PRECISE = "PRECISE"
    IGNORE = "IGNORE"


@dataclass
class StateInfo:
    identifier: str
    type: StateType
    success: StateSuccess = StateSuccess.IGNORE
    space: StateSpace = StateSpace.UNUSED
    min: Union[float, np.ndarray] = None
    max: Union[float, np.ndarray] = None

    def __post_init__(self):
        if self.min is None:
            self.min = np.array([-1.0, -1.0, -1.0])
        if self.max is None:
            self.max = np.array([1.0, 1.0, 1.0])

        # Convert lists to numpy arrays
        if isinstance(self.min, list):
            self.min = np.array(self.min)
        if isinstance(self.max, list):
            self.max = np.array(self.max)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


@dataclass
class TaskInfo:
    precondition: Dict[str, Union[str, float, np.ndarray]]
    space: TaskSpace = TaskSpace.SMALL
    reversed: bool = False
    ee_tp_start: np.ndarray = None
    obj_start: np.ndarray = None
    ee_hrl_start: np.ndarray = None

    def __post_init__(self):
        # Convert lists to numpy arrays and set defaults
        if self.ee_tp_start is None:
            self.ee_tp_start = np.array(
                [
                    0.02586799,
                    -0.23131344,
                    0.57128022,
                    0.73157951,
                    0.68112164,
                    0.02806045,
                    0.00879429,
                ]
            )
        elif isinstance(self.ee_tp_start, list):
            self.ee_tp_start = np.array(self.ee_tp_start)

        if self.obj_start is None:
            self.obj_start = np.array(
                [-0.00699564, 0.40082628, -0.03604347, 0.0, 0.0, 0.0, 1.0]
            )
        elif isinstance(self.obj_start, list):
            self.obj_start = np.array(self.obj_start)

        if self.ee_hrl_start is None:
            self.ee_hrl_start = self.ee_tp_start.copy()
        elif isinstance(self.ee_hrl_start, list):
            self.ee_hrl_start = np.array(self.ee_hrl_start)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class ConfigManager:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self._states = {}
        self._tasks = {}
        self._load_states()
        self._load_tasks()

    def _load_states(self):
        """Load states from JSON config"""
        for state_name, state_data in self.config["states"].items():
            state_info = StateInfo(
                identifier=state_data["identifier"],
                type=StateType(state_data["type"]),
                success=StateSuccess(state_data["success"]),
                space=StateSpace[state_data["space"]],
                min=state_data["min"],
                max=state_data["max"],
            )
            self._states[state_name] = state_info

    def _load_tasks(self):
        """Load tasks from JSON config"""
        for task_name, task_data in self.config["tasks"].items():
            # Process preconditions - convert "min"/"max" strings to actual values
            processed_preconditions = {}
            for state_name, condition in task_data["precondition"].items():
                if isinstance(condition, str):
                    if condition == "min":
                        processed_preconditions[state_name] = self._states[
                            state_name
                        ].min
                    elif condition == "max":
                        processed_preconditions[state_name] = self._states[
                            state_name
                        ].max
                    else:
                        processed_preconditions[state_name] = condition
                else:
                    processed_preconditions[state_name] = condition

            task_info = TaskInfo(
                precondition=processed_preconditions,
                space=TaskSpace[task_data["space"]],
                reversed=task_data["reversed"],
                ee_tp_start=task_data["ee_tp_start"],
                obj_start=task_data["obj_start"],
                ee_hrl_start=task_data["ee_hrl_start"],
            )
            self._tasks[task_name] = task_info

    def get_state(self, name: str) -> StateInfo:
        """Get state by name"""
        return self._states[name]

    def get_task(self, name: str) -> TaskInfo:
        """Get task by name"""
        return self._tasks[name]

    def get_all_states(self) -> Dict[str, StateInfo]:
        """Get all states"""
        return self._states.copy()

    def get_all_tasks(self) -> Dict[str, TaskInfo]:
        """Get all tasks"""
        return self._tasks.copy()

    def from_string(self, name: str) -> str:
        """Find state name by identifier"""
        for state_name, state_info in self._states.items():
            if state_info.identifier in name:
                return state_name
        raise NotImplementedError(f"State for identifier '{name}' does not exist.")

    def get_tp_by_index(self, index: int) -> tuple[str, str]:
        """Get transform and quaternion state names by index"""
        state_names = list(self._states.keys())
        return (state_names[index], state_names[index + 10])

    def get_task_by_index(self, index: int) -> str:
        """Get task name by index"""
        return list(self._tasks.keys())[index]

    def convert_to_states(self, state_space: StateSpace) -> List[str]:
        """Convert state space enum to list of state names"""
        states = []
        for state_name, state_info in self._states.items():
            if state_info.space.value <= state_space.value:
                states.append(state_name)
        return states

    def convert_to_tasks(self, task_space: TaskSpace) -> List[str]:
        """Convert task space enum to list of task names"""
        tasks = []
        for task_name, task_info in self._tasks.items():
            if task_info.space.value <= task_space.value:
                tasks.append(task_name)
        return tasks


# Global config manager instance
_config_manager = None


def get_config_manager(config_path: str = None) -> ConfigManager:
    """Get the global config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


# Backward compatibility functions
def convert_to_states(state_space: StateSpace) -> List[str]:
    return get_config_manager().convert_to_states(state_space)


def convert_to_tasks(task_space: TaskSpace) -> List[str]:
    return get_config_manager().convert_to_tasks(task_space)
