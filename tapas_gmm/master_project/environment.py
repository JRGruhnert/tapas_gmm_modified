from dataclasses import dataclass

import numpy as np
import torch

from tapas_gmm.env.calvin import Calvin
from tapas_gmm.master_project.state import (
    State,
    StateIdent,
    StateSpace,
)
from tapas_gmm.master_project.task import (
    TaskIdent,
    TaskSpace,
    Task,
)
from tapas_gmm.master_project.observation import MasterObservation, make_tapas_format
from tapas_gmm.master_project.evaluator import (
    StateEvaluator,
    EvaluatorConfig,
)
from tapas_gmm.master_project.sampler import SceneMaker


@dataclass
class MasterEnvConfig:
    task_space: TaskSpace
    state_space: StateSpace
    eval_mode: bool
    pybullet_vis: bool
    debug_vis: bool
    evaluator: EvaluatorConfig
    real_time: bool


class MasterEnv:
    def __init__(
        self,
        config: MasterEnvConfig,
        states: list[State],
        tasks: list[Task],
    ):
        self.config = config
        self.env = Calvin(
            eval=config.eval_mode,
            vis=config.pybullet_vis,
            real_time=config.real_time,
        )
        self.evaluator = StateEvaluator(
            config.evaluator,
            self.env.surfaces,
            states,
            tasks,
        )
        self.scene_maker = SceneMaker(states)
        self.current: MasterObservation = None
        self.last_gripper_action = [1.0]  # open

    def reset(self) -> tuple[MasterObservation, MasterObservation]:
        sample, _, _, _ = self.env.reset(settle_time=0)
        ### Sampling goal and current state
        start_scene, goal_scene = self.scene_maker.make(sample.scene_obs)
        # Reset environment twice to get CalvinObservation (maybe find a better way)
        # NOTE do not switch order of resets!!
        calvin_goal, _, _, _ = self.env.reset(goal_scene, static=False, settle_time=50)
        self.calvin_current, _, _, _ = self.env.reset(
            start_scene, static=False, settle_time=50
        )
        self.current = MasterObservation(self.calvin_current)
        self.goal = MasterObservation(calvin_goal)
        self.evaluator.reset(self.current, self.goal)
        return self.current, self.goal

    def step(
        self, task: Task, verbose: bool = False
    ) -> tuple[float, bool, MasterObservation]:
        viz_dict = {}  # TODO: Make available
        task.policy.reset_episode(self.env)
        # Batch prediction for the given observation
        try:
            prediction, _ = task.policy.predict(
                make_tapas_format(self.calvin_current, task, self.goal)
            )
            for action in prediction:
                if len(action.gripper) is not 0:
                    self.last_gripper_action = action.gripper
                ee_action = np.concatenate(
                    (
                        action.ee,
                        self.last_gripper_action,
                    )
                )
                self.calvin_current, _, _, _ = self.env.step(
                    ee_action, self.config.debug_vis, viz_dict
                )
                if verbose:
                    print(self.calvin_current.ee_pose)
                    print(self.calvin_current.ee_state)
                self.current = MasterObservation(self.calvin_current)
        except FloatingPointError:
            # At some point the model crashes.
            # Have to debug if its because of bad input but seems to be not relevant for training
            print(f"Error happened!")
        reward, done = self.evaluator.evaluate(self.current)
        return reward, done, self.current

    def close(self):
        self.env.close()
