from enum import Enum
import torch


class StateSpace(Enum):
    SMALL = 0
    ALL = 1
    UNUSED = 2


class StateType(Enum):
    Euler = 0
    Quat = 1
    Scalar = 2


class StateSuccess(Enum):
    AREA = 0  # Has to be in area
    PRECISE = 1  # Has to be precise (with threshold)
    IGNORE = 2  # Is not used in Success calculation
    # (mostly quaternions cause i am not able to make models that precise)


class StateIdent(Enum):
    @classmethod
    def by_name(cls, name: str):
        """Get enum member by its actual enum member name"""
        try:
            return cls[name]
        except KeyError:
            raise ValueError(f"No StateIdent with name '{name}'")

    ee_euler = 0
    ee_quat = 1
    ee_state = 2
    base__slide_euler = 3
    base__slide_quat = 4
    base__slide = 5
    base__drawer_euler = 6
    base__drawer_quat = 7
    base__drawer = 8
    base__button_euler = 9
    base__button_quat = 10
    base__button = 11
    base__led_euler = 12
    base__led_quat = 13
    base__led = 14
    block_red_euler = 15
    block_red_quat = 16
    block_red = 17
    block_blue_euler = 18
    block_blue_quat = 19
    block_blue = 20
    block_pink_euler = 21
    block_pink_quat = 22
    block_pink = 23
    base__switch_euler = 24  # UNUSED CAUSE NOT ABLE TO RECORD
    base__switch_quat = 25  # UNUSED CAUSE NOT ABLE TO RECORD
    base__switch = 26  # UNUSED CAUSE NOT ABLE TO RECORD
    base__lightbulb_euler = 27  # UNUSED CAUSE NOT ABLE TO RECORD
    base__lightbulb_quat = 28  # UNUSED CAUSE NOT ABLE TO RECORD
    base__lightbulb = 29  # UNUSED CAUSE NOT ABLE TO RECORD


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
    def from_json(cls, json_data: dict) -> "State":
        raise NotImplementedError(
            "This method should be implemented in subclasses to handle JSON deserialization."
        )

    @classmethod
    def convert_to_states(state_space: StateSpace) -> list["State"]:
        raise NotImplementedError(
            "This method should be implemented in subclasses to handle state conversion."
        )

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
