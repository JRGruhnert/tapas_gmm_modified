from conf.master.shared.experiment import Exp1Config
from tapas_gmm_modified.master_project.networks import NetworkType


config = Exp1Config(
    nt=NetworkType.BASELINE_V1,
    pe=0.0,
    pr=0.0,
    workers=1,
)
