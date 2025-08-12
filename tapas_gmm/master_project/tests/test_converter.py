import numpy as np
import pytest

from tapas_gmm.master_project.converter import (
    QuaternionConverter,
    ScalarConverter,
    TransformConverter,
)
from tapas_gmm.master_project.state import StateType
from tapas_gmm.master_project.state import StateIdent

normalized = True


@pytest.mark.parametrize(
    "state, value, goal, exp_value, exp_dist",
    [
        (StateIdent.EE_State, 1.0, -1.0, 1.0, 1.0),
        (StateIdent.EE_State, -1.0, -1.0, 0.0, 0.0),
        (StateIdent.EE_State, 1.0, 1.0, 1.0, 0.0),
        (StateIdent.EE_State, -1.0, 1.0, 0.0, 1.0),
        (StateIdent.Slide_State, -1.0, 0.1, 0.0, 0.0),
    ],
)
def test_converter(
    state: StateIdent,
    value: float | np.ndarray,
    goal: float | np.ndarray,
    exp_value: float | np.ndarray,
    exp_dist: float | np.ndarray,
) -> None:
    if state.value.type == StateType.Scalar:
        converter = ScalarConverter(state=state, normalized=normalized)
        assert converter.value(value) == exp_value
        assert converter.distance(value, goal) == exp_dist
    elif state.value.type in [StateType.Transform, StateType.Quat]:
        converter = TransformConverter(state=state, normalized=normalized)
        assert converter.value(value) == exp_value
        assert converter.distance(value, goal) == exp_dist
    else:
        converter = QuaternionConverter(state=state, normalized=normalized)
        assert converter.value(value) == exp_value
        assert converter.distance(value, goal) == exp_dist
