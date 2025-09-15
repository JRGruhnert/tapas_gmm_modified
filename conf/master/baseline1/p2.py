from conf.master.shared.agent import experiment1
from conf.master.shared.train_env import p2_baseline
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    tag="p2",
    nt=NetworkType.BASELINE_V1,
    env=p2_baseline,
    agent=experiment1,
    state_space=StateSpace.Normal,
    task_space=TaskSpace.Normal,
)
