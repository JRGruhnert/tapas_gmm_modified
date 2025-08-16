from dataclasses import dataclass
from omegaconf import OmegaConf, SCMode

from tapas_gmm.master_project.environment import MasterEnv, MasterEnvConfig
from tapas_gmm.master_project.agent import AgentConfig
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_project.state import State, StateSpace
from tapas_gmm.master_project.task import Task
from tapas_gmm.utils.argparse import parse_and_build_config


@dataclass
class MasterConfig:
    task_space: StateSpace
    state_space: StateSpace
    tag: str
    nt: NetworkType
    agent: AgentConfig
    env: MasterEnvConfig
    verbose: bool = True


def train_agent(config: MasterConfig):
    # Initialize the environment and agent
    task1 = Task(
        "PressButton", reversed=False, conditional=False, overrides=[]
    )  # Example task
    task2 = Task(
        "PressButton", reversed=True, conditional=False, overrides=[]
    )  # Example task
    task3 = Task(
        "OpenDrawer", reversed=False, conditional=False, overrides=[]
    )  # Example task
    task4 = Task(
        "OpenDrawer", reversed=True, conditional=False, overrides=[]
    )  # Example task
    task5 = Task(
        "CloseDrawer", reversed=False, conditional=False, overrides=[]
    )  # Example task
    task6 = Task(
        "CloseDrawer", reversed=True, conditional=False, overrides=[]
    )  # Example task
    task7 = Task(
        "OpenDoor", reversed=False, conditional=False, overrides=[]
    )  # Example task
    task8 = Task(
        "OpenDoor", reversed=True, conditional=False, overrides=[]
    )  # Example task
    task9 = Task(
        "CloseDoor", reversed=False, conditional=False, overrides=[]
    )  # Example task
    task10 = Task(
        "CloseDoor", reversed=True, conditional=False, overrides=[]
    )  # Example task
    task11 = Task(
        "GrabBlueDrawer", reversed=False, conditional=False, overrides=[]
    )  # Example task
    task12 = Task(
        "GrabBlueDrawer", reversed=True, conditional=False, overrides=[]
    )  # Example task
    tasks = [
        task1,
        task2,
        task3,
        task4,
        task5,
        task6,
        task7,
        task8,
        task9,
        task10,
        task11,
        task12,
    ]
    states = State.from_json_list(config.state_space)
    env = MasterEnv(config.env, states, tasks)
    task5.initialize_task_parameters(states)
    print(f"Task parameters: {task5.task_parameters}")
    env.reset()
    while True:
        # TODO ask in console about next task. preferable via index with name of task
        print(f"{0}: Reset")
        for i, task in enumerate(tasks, start=1):
            print(f"{i}: {task.name}")
        choice = input("Enter the Task id: ")
        task_id = int(choice)
        if task_id == 0:
            print("Resetting environment...")
            env.reset()
        else:
            task_id -= 1  # Adjust for zero-based index
            env.step(tasks[task_id], verbose=True)


def entry_point():

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    train_agent(config)


if __name__ == "__main__":
    entry_point()
