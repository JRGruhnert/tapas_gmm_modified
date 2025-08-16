from tapas_gmm.master_project.environment import MasterEnvConfig
from conf.env.calvin.env_eval_conf import calvin_env_config

env = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=False,
)

test_env = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=True,
)
