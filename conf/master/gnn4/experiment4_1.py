from conf.master.shared.train_env import env
from conf.master.shared.agent import experiment1
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from tapas_gmm.master_retrain import MasterConfig


config = MasterConfig(
    state_space=StateSpace.Normal,
    task_space=TaskSpace.Normal,
    verbose=True,
    tag="normal_normal_retrain",
    nt=NetworkType.GNN_V4,
    env=env,
    agent=experiment1,
    checkpoint="results/gnn4/min_min/model_cp_best.pth",
    keep_epoch=False,  # Keep the epoch number in the checkpoint
)
