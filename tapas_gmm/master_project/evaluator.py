from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
from tapas_gmm.master_project.state import State, StateIdent, StateSuccess, StateType
from tapas_gmm.master_project.task import Task
from tapas_gmm.master_project.observation import MasterObservation


class RewardMode(Enum):
    SPARSE = 0
    RANGE = 1
    ONOFF = 2


@dataclass
class EvaluatorConfig:
    allowed_steps: int = 18
    reward_mode: RewardMode = RewardMode.SPARSE
    max_reward: float = 100.0
    min_reward: float = 0.0
    success_threshold: float = 0.05


class StateEvaluator:
    def __init__(
        self,
        config: EvaluatorConfig,
        surfaces: dict[str, np.ndarray],
        states: list[State],
        tasks: list[Task],
    ):
        self.config = config
        self.steps_left = 20  # TODO: Change
        self.states = states
        self.tasks = tasks
        self.surfaces = surfaces
        # Internal States
        self.last: MasterObservation = None
        self.goal: MasterObservation = None
        self.terminal = False

    def reset(self, obs: MasterObservation, goal: MasterObservation):
        self.steps_left = 20  # TODO: Change
        self.last = obs
        self.goal = goal
        self.terminal = False

    def evaluate(self, obs: MasterObservation) -> tuple[float, bool]:
        if self.terminal:
            raise UserWarning(
                "Episode already ended. Please reset the evaluator with the new goal and state."
            )
        self.steps_left -= 1
        if self.config.reward_mode is RewardMode.SPARSE:
            terminal = self.is_terminal(obs)
            if terminal:  # Success
                self.terminal = terminal
                # print(f"success {self.steps_left}")
                return self.config.max_reward, self.terminal
            else:
                if self.steps_left == 0:  # Failure
                    self.terminal = True
                    # print(f"failure {self.steps_left}")
                    return self.config.min_reward, self.terminal
                else:  # Normal Step
                    self.terminal = False
                    # print(f"normal {self.steps_left}")
                    return self.config.min_reward, self.terminal
        if self.config.reward_mode is RewardMode.ONOFF:
            raise NotImplementedError("Reward Mode not implemented.")
        if self.config.reward_mode is RewardMode.RANGE:
            raise NotImplementedError("Reward Mode not implemented.")

    def difference_reward(
        self,
        prev: dict[StateIdent, torch.Tensor],
        next: dict[StateIdent, torch.Tensor],
    ) -> float:
        total = 0.0
        for key in prev:
            diff = prev[key].item() - next[key].item()
            scale = self.normalized_reward_scale[key.value.type]
            # Multiply and accumulate
            total += scale * diff

        return total / len(prev)

    def on_off_reward(
        self,
        prev: dict[StateIdent, torch.Tensor],
        next: dict[StateIdent, torch.Tensor],
    ) -> float:
        total = 0.0
        for key in prev:
            diff = prev[key].item() - next[key].item()
            scale = self.normalized_reward_scale[key.value.type]
            if diff < 0:
                total -= 1.0 * scale
            elif diff > 0:
                total += 1.0 * scale
        return total / len(prev)

    def is_terminal(
        self,
        obs: MasterObservation,
    ) -> bool:
        ##### Checking if goal is reached
        goal_reached = True
        for state in self.states:
            state_type = state.type
            if state.success == StateSuccess.Area:
                if state_type is StateType.Quat or state_type is StateType.Scalar:
                    raise ValueError(
                        "Quaternion and Scalar States don't support area based evaluation."
                    )
                goal_value = self.goal.states[state]
                goal_surface = self.check_surface(goal_value)
                next_value = obs.states[state]
                next_surface = self.check_surface(next_value)
                if goal_surface != next_surface:
                    goal_reached = False
                    break
            elif state.success == StateSuccess.Precise:
                if (
                    state.distance(
                        obs.states[state.ident], self.goal.states[state.ident]
                    ).item()
                    > self.config.success_threshold
                ):
                    goal_reached = False
                    break
            elif state.success == StateSuccess.Ignore:
                pass  # Probably only Quaternions cause of Model recording
                # State is not evaluated
            else:
                raise NotImplementedError(
                    f"State Success type: {state.success} is not implemented."
                )
        return goal_reached

    def check_surface(self, transform) -> str | None:
        for name, (min_corner, max_corner) in self.surfaces.items():
            box_min = np.array(min_corner)
            box_max = np.array(max_corner)
            if np.all(transform >= box_min) and np.all(transform <= box_max):
                return name
        # sampling_range = np.array(self.surfaces["test"])  # TODO: Change
        return None
