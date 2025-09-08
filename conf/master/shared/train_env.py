from tapas_gmm.master_project.environment import MasterEnvConfig
from conf.env.calvin.env_eval_conf import calvin_env_config

env = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=False,
)

p3_baseline = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=False,
    # p_empty=0.0,
    # p_rand=0.0,
)


p3_gnn = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=False,
    # p_empty=0.0,
    # p_rand=0.0,
)

env_baseline = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=False,
    p_empty=0.0,
    p_rand=0.2,
)

env_gnn = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=False,
    p_empty=0.0,
    p_rand=0.2,
)


re_env_gnn = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=False,
)

re_env_baseline = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=False,
)


debug_env = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=True,
    # p_empty=0.0,
    # p_rand=0.0,
)

eval_env = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=False,
    # p_empty=0.0,
    # p_rand=0.0,
)
