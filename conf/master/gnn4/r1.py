from conf.master.shared.train_env import re_env_gnn
from conf.master.shared.agent import experiment1
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from tapas_gmm.master_retrain import RetrainConfig


config = RetrainConfig(
    state_space=StateSpace.Normal,
    task_space=TaskSpace.Normal,
    verbose=True,
    tag="r1",
    nt=NetworkType.GNN_V4,
    env=re_env_gnn,
    agent=experiment1,
    checkpoint="results/gnn4/p1_pe_0.0_pr_0.0/model_cp_best.pth",
    keep_epoch=False,  # Keep the epoch number in the checkpoint
)
