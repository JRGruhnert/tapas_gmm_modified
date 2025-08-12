from build.lib.tapas_gmm.master_project.definitions import TaskSpace
from tapas_gmm.master_project.storage import PolicyStorageConfig


minimal_policies = PolicyStorageConfig(
    storage_path="data/storage",
    task_space=TaskSpace.MINIMAL,
)


all_policies = PolicyStorageConfig(
    storage_path="data/storage",
    task_space=TaskSpace.ALL,
)
