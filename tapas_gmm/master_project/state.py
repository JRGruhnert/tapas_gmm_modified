from enum import Enum
import torch
import json
from pathlib import Path
from typing import List


class StateSpace(Enum):
    Minimal = "Minimal"
    All = "All"
    Unused = "Unused"


class StateType(Enum):
    Euler = "Euler"
    Quat = "Quat"
    Scalar = "Scalar"


class StateSuccess(Enum):
    Area = "Area"  # Has to be in area
    Precise = "Precise"  # Has to be precise (with threshold)
    Ignore = "Ignore"  # Is not used in Success calculation


class StateIdent(Enum):
    EE_Euler = "ee_euler"
    EE_Quat = "ee_quat"
    EE_State = "ee_state"
    Slide_Euler = "base__slide_euler"
    Slide_Quat = "base__slide_quat"
    Slide_State = "base__slide"
    Drawer_Euler = "base__drawer_euler"
    Drawer_Quat = "base__drawer_quat"
    Drawer_State = "base__drawer"
    Button_Euler = "base__button_euler"
    Button_Quat = "base__button_quat"
    Button_State = "base__button"
    Led_Euler = "led_euler"
    Led_Quat = "led_quat"
    Led_State = "led"
    Block_Red_Euler = "block_red_euler"
    Block_Red_Quat = "block_red_quat"
    Block_Red_State = "block_red"
    Block_Blue_Euler = "block_blue_euler"
    Block_Blue_Quat = "block_blue_quat"
    Block_Blue_State = "block_blue"
    Block_Pink_Euler = "block_pink_euler"
    Block_Pink_Quat = "block_pink_quat"
    Block_Pink_State = "block_pink"
    Switch_Euler = "base__switch_euler"  # UNUSED CAUSE NOT ABLE TO RECORD
    Switch_Quat = "base__switch_quat"  # UNUSED CAUSE NOT ABLE TO RECORD
    Switch_State = "base__switch"  # UNUSED CAUSE NOT ABLE TO RECORD
    Lightbulb_Euler = "lightbulb_euler"  # UNUSED CAUSE NOT ABLE TO RECORD
    Lightbulb_Quat = "lightbulb_quat"  # UNUSED CAUSE NOT ABLE TO RECORD
    Lightbulb_State = "lightbulb"  # UNUSED CAUSE NOT ABLE TO RECORD


class State:
    def __init__(
        self,
        ident: StateIdent,
        type: StateType,
        success: StateSuccess,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
    ):
        self._ident = ident
        self._type = type
        self._success = success
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    @classmethod
    def from_json(cls, ident_value: str, json_data: dict) -> "State":
        """Create a State instance from JSON data"""
        if (
            "type" not in json_data
            or "space" not in json_data
            or "success" not in json_data
            or "lower_bound" not in json_data
            or "upper_bound" not in json_data
        ):
            raise ValueError(f"Invalid JSON data for State {ident_value}")
        if not isinstance(json_data["lower_bound"], list):
            raise ValueError(f"Invalid JSON data for State {ident_value}")
        if not isinstance(json_data["upper_bound"], list):
            raise ValueError(f"Invalid JSON data for State {ident_value}")
        return cls(
            ident=StateIdent(ident_value),
            type=StateType(json_data["type"]),
            success=StateSuccess(json_data["success"]),
            lower_bound=torch.tensor(json_data["lower_bound"], dtype=torch.float32),
            upper_bound=torch.tensor(json_data["upper_bound"], dtype=torch.float32),
        )

    @classmethod
    def from_json_list(cls, state_space: StateSpace) -> List["State"]:
        """Convert a StateSpace to a list of State objects by reading from states.json"""
        # Load the states.json file
        states_json_path = Path(__file__).parent / "data" / "states.json"

        if not states_json_path.exists():
            raise FileNotFoundError(f"States JSON file not found at {states_json_path}")

        with open(states_json_path, "r") as f:
            states_data = json.load(f)

        # Filter states based on the requested state space
        filtered_states = []

        for ident, state_data in states_data.items():
            # Check if this state belongs to the requested space
            state_space_str = state_data.get("space", "Unused")

            # Apply filtering logic based on StateSpace hierarchy:
            # SMALL (0) includes only SMALL states
            # ALL (1) includes SMALL + ALL states
            # UNUSED (2) includes only UNUSED states
            include_state = False

            if state_space == StateSpace.Minimal and state_space_str == "Minimal":
                include_state = True
            elif state_space == StateSpace.All and state_space_str in [
                "Minimal",
                "All",
            ]:
                include_state = True
            elif state_space == StateSpace.Unused:
                raise ValueError(
                    "Unused space is not supported in convert_to_states method"
                )

            if include_state:
                state = cls.from_json(ident, state_data)
                filtered_states.append(state)

        return filtered_states

    @property
    def ident(self) -> StateIdent:
        """Returns the StateIdent of the state."""
        return self._ident

    @property
    def type(self) -> StateType:
        """Returns the StateType of the state."""
        return self._type

    @property
    def success(self) -> StateSuccess:
        """Returns the StateSuccess of the state."""
        return self._success

    @property
    def lower_bound(self) -> torch.Tensor:
        """Returns the lower bound of the state."""
        return self._lower_bound

    @property
    def upper_bound(self) -> torch.Tensor:
        """Returns the upper bound of the state."""
        return self._upper_bound

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize a value x to the range [0, 1] based on min and max.
        """
        return (x - self.lower_bound) / (self.upper_bound - self.lower_bound)

    def normalize_quat(self, q: torch.Tensor) -> torch.Tensor:
        return q / torch.norm(q)

    def canonicalize_quat(self, q: torch.Tensor) -> torch.Tensor:
        """
        Enforce qw >= 0 by flipping sign if needed.
        q is assumed unit‑length.
        """
        # quaternion format [qx, qy, qz, qw]
        if q[3] < 0:
            return -q
        return q

    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Clamp a value x to the range defined by lower_bound and upper_bound.
        """
        return torch.clamp(x, self.lower_bound, self.upper_bound)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the value of the state as a tensor."""
        if self.type == StateType.Euler:
            cx = self.clamp(x)
            return self.normalize(cx)
        elif self.type == StateType.Quat:
            nx = self.normalize_quat(x)
            return self.canonicalize_quat(nx)
        elif self.type == StateType.Scalar:
            nx = self.clamp(x)
            return self.normalize(nx)
        else:
            raise ValueError(f"Unknown state type: {self.type}")

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Returns the distance of the state as a tensor."""
        if self.type == StateType.Euler:
            cx = self.clamp(x)
            cy = self.clamp(y)
            nx = self.normalize(cx)
            ny = self.normalize(cy)
            return torch.norm(nx - ny)
        elif self.type == StateType.Quat:
            cx = self.normalize_quat(x)
            cy = self.normalize_quat(y)
            dot = torch.clamp(torch.abs(torch.dot(cx, cy)), -1.0, 1.0)
            return 2.0 * torch.arccos(dot)
        elif self.type == StateType.Scalar:
            cx = self.clamp(x)
            cy = self.clamp(y)
            nx = self.normalize(cx)
            ny = self.normalize(cy)
            return torch.abs(nx - ny)
        else:
            raise ValueError(f"Unknown state type: {self.type}")
