import pathlib
import time
from dataclasses import dataclass
from typing import Any
import numpy as np
from omegaconf import DictConfig, OmegaConf, SCMode
from loguru import logger
from tqdm.auto import tqdm

from tapas_gmm.collect_data import Config
from tapas_gmm.master_project.environment import MasterEnv, MasterEnvConfig
from tapas_gmm.master_project.state import State, StateSpace
from tapas_gmm.master_project.task import Task, TaskSpace
from tapas_gmm.policy import PolicyEnum
from tapas_gmm.policy.manual import ManualPolicy
from tapas_gmm.dataset.scene import SceneDataset, SceneDatasetConfig
from tapas_gmm.utils.keyboard_observer import KeyboardObserver
from tapas_gmm.utils.argparse import parse_and_build_config
from tapas_gmm.utils.misc import (
    DataNamingConfig,
    get_dataset_name,
    loop_sleep,
)


@dataclass
class Config:

    task: str
    n_episodes: int
    sequence_len: int | None

    data_naming: DataNamingConfig
    dataset_config: SceneDatasetConfig

    policy_type: PolicyEnum
    policy: Any
    state_space: StateSpace
    task_space: TaskSpace
    env: MasterEnvConfig
    horizon: int | None = None

    pretraining_data: bool = False


def main(config: Config) -> None:
    states = State.from_json_list(config.state_space)
    env = MasterEnv(config=config.env, states=states, tasks=[])
    keyboard_obs = KeyboardObserver()
    policy = ManualPolicy(config, env, keyboard_obs)

    assert config.data_naming.data_root is not None

    save_path = pathlib.Path(config.data_naming.data_root) / config.task

    if not save_path.is_dir():
        logger.warning(
            "Creating save path. This should only be needed for " "new tasks."
        )
        save_path.mkdir(parents=True)

    replay_memory = SceneDataset(
        allow_creation=True,
        config=config.dataset_config,
        data_root=save_path / config.data_naming.feedback_type,
    )

    obs = env.wrapped_reset()

    time.sleep(5)
    logger.info("Go!")

    episodes_count = 0
    timesteps = 0

    # Max number of timesteps in an episode
    horizon = config.horizon or np.inf

    try:
        with tqdm(total=config.n_episodes) as ebar:
            with tqdm(total=horizon) as tbar:
                tbar.set_description("Time steps")
                while episodes_count < config.n_episodes:
                    ebar.set_description("Running episode")
                    start_time = time.time()

                    prediction, policy_done, policy_success = policy.predict(
                        obs,
                        True,
                    )  # Action is relative
                    try:
                        obs = env.wrapped_direct_step(prediction, verbose=True)
                    except RuntimeError as e:
                        logger.error(f"Raw action: {prediction}")
                        logger.error(f"Error: {e}")
                        raise e
                    replay_memory.add_observation(obs)

                    timesteps += 1
                    tbar.update(1)

                    if policy_done and policy_success:
                        # logger.info("Saving trajectory.")
                        ebar.set_description("Saving trajectory")
                        replay_memory.save_current_traj()
                        obs = env.wrapped_reset()
                        keyboard_obs.reset()
                        policy.reset_episode(env)

                        episodes_count += 1
                        ebar.update(1)

                        timesteps = 0
                        tbar.reset()

                    elif policy_done and not policy_success or timesteps >= horizon:
                        # logger.info("Resetting without saving traj.")
                        ebar.set_description("Resetting without saving traj")
                        replay_memory.reset_current_traj()
                        obs = env.wrapped_reset()
                        keyboard_obs.reset()
                        policy.reset_episode(env)

                        timesteps = 0
                        tbar.reset()

                    else:
                        loop_sleep(start_time)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ...")
        env.close()


def complete_config(config: DictConfig) -> DictConfig:
    config.env.calvin_config.task = config.data_naming.task
    config.task = config.data_naming.task
    config.dataset_config.data_root = config.data_naming.data_root
    config.data_naming.feedback_type = get_dataset_name(config)
    return config


def entry_point():
    _, dict_config = parse_and_build_config(data_load=False, extra_args={})
    dict_config = complete_config(dict_config)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    main(config)  # type: ignore


if __name__ == "__main__":
    entry_point()
