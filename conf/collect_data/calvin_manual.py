from omegaconf import MISSING
from conf._machine import data_naming_config
from conf.dataset.scene.calvin import scene_dataset_config
from conf.master.shared.collect_env import collect_env
from tapas_gmm.tapas_collect import Config
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from tapas_gmm.policy import PolicyEnum

config = Config(
    task=MISSING,
    n_episodes=5,
    sequence_len=None,
    data_naming=data_naming_config,
    dataset_config=scene_dataset_config,
    env=collect_env,
    state_space=StateSpace.Normal,
    task_space=TaskSpace.Normal,
    policy_type=PolicyEnum.MANUAL,
    policy=None,
)
