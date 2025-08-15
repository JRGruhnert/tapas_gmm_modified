from tapas_gmm.master_project.task import TaskSpace
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.environment import MasterEnvConfig
from conf.env.calvin.env_eval_conf import calvin_env_config

env1 = MasterEnvConfig(
    task_space=TaskSpace.Minimal,
    state_space=StateSpace.Minimal,
    calvin_config=calvin_env_config,
    debug_vis=False,
)

env2 = MasterEnvConfig(
    task_space=TaskSpace.Minimal,
    state_space=StateSpace.All,
    calvin_config=calvin_env_config,
    debug_vis=False,
)

env3 = MasterEnvConfig(
    task_space=TaskSpace.All,
    state_space=StateSpace.All,
    calvin_config=calvin_env_config,
    debug_vis=False,
)

test_env = MasterEnvConfig(
    task_space=TaskSpace.All,
    state_space=StateSpace.All,
    calvin_config=calvin_env_config,
    debug_vis=False,
)
