from conf.master.shared.train_env import test_env
from conf.master.shared.agent import test
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    task_space=TaskSpace.All,  # Assuming task_space is not used in this context
    state_space=StateSpace.All,  # Assuming state_space is not used in this context
    tag="test",
    nt=NetworkType.BASELINE_TEST,
    env=test_env,
    agent=test,
)
