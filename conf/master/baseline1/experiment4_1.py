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
    tag="r1",
    nt=NetworkType.BASELINE_V1,
    env=env,
    agent=experiment1,
    checkpoint="results/baseline1/p1_pe_0.0_pr_0.0/model_cp_best.pth",
    keep_epoch=False,  # Keep the epoch number in the checkpoint
)
