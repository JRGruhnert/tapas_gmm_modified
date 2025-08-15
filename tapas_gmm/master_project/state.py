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
    Euler_Angle = "Euler"
    Axis_Angle = "Axis"
    Quaternion = "Quaternion"
    Range = "Range"
    Boolean = "Boolean"
    Flip = "Flip"  # Special type for flipping the state, e.g., for boolean states


class StateSuccess(Enum):
    Area = "Area"  # Has to be in the same area
    Precise = "Precise"  # Has to be precise (with threshold)
    Ignore = "Ignore"  # Is not used in Success calculation


class State:
    def __init__(
        self,
        name: str,
        type: StateType,
        success: StateSuccess,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
    ):
        self._name = name
        self._type = type
        self._success = success
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    @classmethod
    def from_json(cls, name: str, json_data: dict) -> "State":
        """Create a State instance from JSON data"""
        if (
            "type" not in json_data
            or "space" not in json_data
            or "success" not in json_data
            or "lower_bound" not in json_data
            or "upper_bound" not in json_data
        ):
            raise ValueError(f"Invalid JSON data for State {name}")
        if not isinstance(json_data["lower_bound"], list):
            raise ValueError(f"Invalid JSON data for State {name}")
        if not isinstance(json_data["upper_bound"], list):
            raise ValueError(f"Invalid JSON data for State {name}")

        state_type = StateType(json_data["type"])
        # sub_type = StateSubType(json_data["sub_type"])

        # Prepare common arguments for all state types
        common_args = {
            "name": name,
            "type": state_type,
            "success": StateSuccess(json_data["success"]),
            "lower_bound": torch.tensor(json_data["lower_bound"], dtype=torch.float32),
            "upper_bound": torch.tensor(json_data["upper_bound"], dtype=torch.float32),
        }
        # Default implementation for base class or when called directly on subclasses
        return cls(**common_args)

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
    def name(self) -> str:
        """Returns the StateIdent of the state."""
        return self._name

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

    @property
    def relative_threshold(self) -> torch.Tensor:
        """Returns the relative threshold for the state."""
        # Calculate the relative threshold as 5% of the range
        return 0.05 * (self.upper_bound - self.lower_bound)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize a value x to the range [0, 1] based on min and max.
        """
        return (x - self.lower_bound) / (self.upper_bound - self.lower_bound)

    def normalize_quat(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.norm(x)
        if x[3] < 0:
            return -x
        return x

    def quaternion_mean(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean quaternion using the eigenvector method.
        quaternions: tensor of shape [N, 4] (x, y, z, w)
        Returns: mean quaternion [4] in (x, y, z, w) format
        """
        # Swap to (w, x, y, z) for computation
        quats = quaternions[:, [3, 0, 1, 2]]
        quats = quats / quats.norm(dim=1, keepdim=True)
        A = quats.t() @ quats
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        mean_quat = eigenvectors[:, -1]
        # Ensure positive scalar part
        if mean_quat[0] < 0:
            mean_quat = -mean_quat
        # Swap back to (x, y, z, w)
        mean_quat_xyzw = mean_quat[[1, 2, 3, 0]]
        return mean_quat_xyzw

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the value of the state as a tensor."""
        if (
            self.type == StateType.Euler_Angle
            or self.type == StateType.Axis_Angle
            or self.type == StateType.Range
        ):
            return self.normalize(x)
        elif self.type == StateType.Quaternion:
            return self.normalize_quat(x)
        elif self.type == StateType.Boolean or self.type == StateType.Flip:
            # No need to normalize or clamp boolean states
            # as they are already in {0, 1} range.
            return x
        else:
            raise NotImplementedError(
                f"Type {self.type} is not implemented yet. Please implement the value method for {self.type} type."
            )

    def distance(
        self,
        current: torch.Tensor,
        tp: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the distance of the state as a tensor."""
        if self.type == StateType.Euler_Angle or self.type == StateType.Axis_Angle:
            nx = self.normalize(current)
            ny = self.normalize(tp)
            return torch.linalg.vector_norm(nx - ny)
        elif self.type == StateType.Quaternion:
            nx = self.normalize_quat(current)
            ny = self.normalize_quat(tp)
            dot = torch.clamp(torch.abs(torch.dot(nx, ny)), -1.0, 1.0)
            return 2.0 * torch.arccos(dot)
        elif self.type == StateType.Range:
            nx = self.normalize(current)
            ny = self.normalize(tp)
            return torch.abs(nx - ny)
        elif self.type == StateType.Boolean:
            return torch.abs(current - tp)
        elif self.type == StateType.Flip:
            return tp - torch.abs(current - goal)  # Flips distance
        else:
            raise ValueError(f"Unknown state type: {self.type}")

    def as_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        object_tps: list[str],
    ) -> tuple[bool, torch.Tensor]:
        """Returns the mean of the given tensor values."""
        if self.name not in object_tps:
            return False, None  # Ignore if not in object_tps
        if self.type == StateType.Euler_Angle or self.type == StateType.Axis_Angle:
            if reversed:
                return True, end.mean(dim=0)
            return True, start.mean(dim=0)
        elif self.type == StateType.Quaternion:
            if reversed:
                return True, self.quaternion_mean(end)
            return True, self.quaternion_mean(start)
        elif self.type == StateType.Range:
            if reversed:
                std = end.std(dim=0)
                if (std < self.relative_threshold).all():
                    return True, end.mean(dim=0)
            else:
                std = start.std(dim=0)
                if (std < self.relative_threshold).all():
                    return True, start.mean(dim=0)
            return False, None  # Not constant enough
        elif self.type == StateType.Boolean:
            if reversed:
                std = end.std(dim=0)
                if (std == 0).all():
                    return True, end.mean(dim=0)
            else:
                std = start.std(dim=0)
                if (std == 0).all():
                    return True, start.mean(dim=0)
            return False, None  # Not constant enough
        elif self.type == StateType.Flip:
            # Check if end is always the opposite of start
            if (end == (1 - start)).all(dim=0).all():
                return True, torch.tensor([1.0])  # Flip state
            return False, None
        else:
            raise ValueError(f"Unknown state type: {self.type}")

    def rotation_mean_exp_map(self, rotations: torch.Tensor) -> torch.Tensor:
        """
        rotations: tensor of shape [N, 4] where each row is [w, x, y, z]
        """
        # Convert rotations to rotation vectors (axis-angle representation)
        rotation_vectors = []

        for q in rotations:
            # Normalize rotation quaternion
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
