from conf.master.shared.train_env import env
from conf.master.shared.agent import experiment1
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    state_space=StateSpace.Normal,
    task_space=TaskSpace.Minimal,
    verbose=False,
    tag="new_normal_min",
    nt=NetworkType.GNN_V4,
    env=env,
    agent=experiment1,
)
