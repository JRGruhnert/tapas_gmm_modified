from conf.master.shared.train_env import p1_gnn
from conf.master.shared.agent import experiment1
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    tag="p1",
    nt=NetworkType.GNN_V4,
    env=p1_gnn,
    agent=experiment1,
    state_space=StateSpace.Minimal,
    task_space=TaskSpace.Minimal,
)
