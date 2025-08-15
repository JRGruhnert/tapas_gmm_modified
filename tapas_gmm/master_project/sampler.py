import itertools
import random
import numpy as np

from tapas_gmm.master_project.state import State, StateIdent, StateType


class SceneMaker:
    def __init__(self, surfaces: dict[str, np.ndarray], states: list[State]):
        self.surfaces = surfaces
        self.states = states

    def make(self, scene_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Samples a new scene observation based on the current one."""
        # Sample a pre-condition
        pre_condition = self._sample_pre_condition(scene_obs)

        # Sample a post-condition that is different from the pre-condition
        post_condition = self._sample_post_condition(pre_condition)

        return pre_condition, post_condition

    def _sample_from_values(self, values):
        return random.choice(values)

    def _parse_scene_obs(self, scene_obs):
        # an object pose is composed of position (3) and orientation (4 for quaternion)  / (3 for euler)
        n_obj = 3
        n_doors = 2
        n_buttons = 1
        n_switches = 1
        n_lights = 2

        split_ids = np.cumsum([n_doors, n_buttons, n_switches, n_lights])
        door_info, button_info, switch_info, light_info, obj_info = np.split(
            scene_obs, split_ids
        )

        assert len(door_info) == n_doors
        assert len(button_info) == n_buttons
        assert len(switch_info) == n_switches
        assert len(light_info) == n_lights
        assert len(obj_info) // n_obj in [
            6,
            7,
        ]  # depending on euler angles or quaternions

        obj_info = np.split(obj_info, n_obj)

        return door_info, button_info, switch_info, light_info, obj_info

    def _update_scene_obs(
        self, scene_dict: dict[StateIdent, np.ndarray | float], scene_obs: np.ndarray
    ) -> np.ndarray:
        """Return state information of the doors, drawers and shelves."""
        door_states, button_states, switch_states, light_states, object_poses = (
            self._parse_scene_obs(scene_obs)
        )

        door_states = [
            scene_dict.get(StateIdent.Slide_State, door_states[0]),
            scene_dict.get(StateIdent.Drawer_State, door_states[1]),
        ]
        button_states = [scene_dict.get(StateIdent.Button_State, button_states[0])]
        switch_states = [scene_dict.get(StateIdent.Switch_State, switch_states[0])]

        light_states = [
            scene_dict.get(StateIdent.Lightbulb_State, light_states[0]),
            scene_dict.get(StateIdent.Led_State, light_states[1]),
        ]

        object_poses = list(
            itertools.chain(
                *[
                    np.concatenate(
                        [
                            scene_dict.get(
                                StateIdent.Block_Red_Euler, object_poses[0][:3]
                            ),
                            scene_dict.get(
                                StateIdent.Block_Red_Quat, object_poses[0][-4:]
                            ),
                        ],
                    ),
                    np.concatenate(
                        [
                            scene_dict.get(
                                StateIdent.Block_Blue_Euler, object_poses[1][:3]
                            ),
                            scene_dict.get(
                                StateIdent.Block_Blue_Quat, object_poses[1][-4:]
                            ),
                        ],
                    ),
                    np.concatenate(
                        [
                            scene_dict.get(
                                StateIdent.Block_Pink_Euler, object_poses[2][:3]
                            ),
                            scene_dict.get(
                                StateIdent.Block_Pink_Quat, object_poses[2][-4:]
                            ),
                        ],
                    ),
                ]
            )
        )

        return np.concatenate(
            [door_states, button_states, switch_states, light_states, object_poses]
        )

    def _sample_pre_condition(self, scene_obs: np.ndarray) -> np.ndarray:
        scene_dict: dict[StateIdent, np.ndarray | float] = {}
        for state in self.states:
            if state.type is StateType.Scalar:
                scene_dict[state.ident] = self._sample_from_values(
                    [state.lower_bound.item(), state.upper_bound.item()]
                )
            elif state.type is StateType.Euler or state.type is StateType.Quat:
                pass  # NOTE: Euler and Quat states are not sampled here, they are set directly
            else:
                raise NotImplementedError("StateType sampling not implemented.")

        if StateIdent.Button_State in scene_dict:
            scene_dict[StateIdent.Led_State] = scene_dict[StateIdent.Button_State]
        if StateIdent.Switch_State in scene_dict:
            scene_dict[StateIdent.Lightbulb_State] = scene_dict[StateIdent.Switch_State]

        return self._update_scene_obs(scene_dict, scene_obs)

    def _sample_post_condition(self, scene_obs: np.ndarray) -> np.ndarray:
        """Samples an environment state that is different to the current one"""

        candidate = self._sample_pre_condition(scene_obs)

        while np.array_equal(candidate, scene_obs):
            candidate = self._sample_pre_condition(scene_obs)

        return candidate
