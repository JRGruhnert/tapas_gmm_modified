from dataclasses import dataclass

from tapas_gmm.master_project.agent import AgentConfig
from tapas_gmm.master_project.environment import MasterEnvConfig
from tapas_gmm.master_project.networks import NetworkType
from conf.env.calvin.env_eval_conf import calvin_env_config
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace


@dataclass
class ExperimentConfig:
    nt: NetworkType
    agent: AgentConfig
    env: MasterEnvConfig
    verbose: bool


@dataclass
class Exp1Config(ExperimentConfig):
    nt: NetworkType  # Set later
    cross_t = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    t_spaces_mapping = {
        1: {"state_space": StateSpace.Minimal, "task_space": TaskSpace.Minimal},
        2: {"state_space": StateSpace.Normal, "task_space": TaskSpace.Minimal},
        3: {"state_space": StateSpace.Normal, "task_space": TaskSpace.Normal},
    }
    p_max: float = 1.0
    p_min: float = 0.0
    p_step: float = 0.1
    pe: float = -1.0  # Negative means no specific
    pr: float = -1.0  # Negative means no specific
    workers: int = 4
    verbose: bool = True
    agent: AgentConfig = AgentConfig()
    env: MasterEnvConfig = MasterEnvConfig(
        calvin_config=calvin_env_config,
        debug_vis=False,
    )
