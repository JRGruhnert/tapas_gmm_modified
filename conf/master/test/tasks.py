from tapas_gmm.master_eval import EvalConfig
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from conf.master.shared.train_env import debug_env
from conf.master.shared.agent import debug

config = EvalConfig(
    task_space=TaskSpace.Normal,  # Assuming task_space is not used in this context
    state_space=StateSpace.Normal,  # Assuming state_space is not used in this context
    tag="test",
    nt=NetworkType.BASELINE_TEST,
    env=debug_env,
    agent=debug,
)
