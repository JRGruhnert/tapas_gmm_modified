from tapas_gmm.master_project.environment import MasterEnvConfig
from conf.env.calvin.env_eval_conf import calvin_env_config

env = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=False,
)

env_negative = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=False,
    min_reward=-1.0,
    p_empty=0.0,
    p_rand=0.5,
)

debug_env = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=True,
)
