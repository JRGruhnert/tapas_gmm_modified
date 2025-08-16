from dataclasses import dataclass
from datetime import datetime
from omegaconf import OmegaConf, SCMode

from tapas_gmm.master_project.state import State, StateSpace
from tapas_gmm.master_project.task import Task, TaskSpace
from tapas_gmm.master_project.environment import MasterEnv, MasterEnvConfig
from tapas_gmm.master_project.agent import MasterAgent, AgentConfig
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.utils.argparse import parse_and_build_config


@dataclass
class MasterConfig:
    state_space: StateSpace
    task_space: TaskSpace
    tag: str
    nt: NetworkType
    agent: AgentConfig
    env: MasterEnvConfig
    verbose: bool = True


def train_agent(config: MasterConfig):
    # Initialize the environment and agent
    tasks = Task.from_json_list(config.task_space)
    states = State.from_json_list(config.state_space)
    env = MasterEnv(config.env, states, tasks)
    agent = MasterAgent(config.agent, config.nt, config.tag, states, tasks)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    batch_start_time = datetime.now().replace(microsecond=0)
    stop_training = False
    while not stop_training:  # Training loop
        terminal = False
        batch_rdy = False
        obs, goal = env.reset()
        while not terminal and not batch_rdy:
            task = agent.act(obs, goal)
            reward, terminal, obs = env.step(task, verbose=config.verbose)
            batch_rdy = agent.feedback(reward, terminal)
        if batch_rdy:
            train_start_time = datetime.now().replace(microsecond=0)
            stop_training = agent.learn(verbose=config.verbose)
            train_end_time = datetime.now().replace(microsecond=0)
            print(
                "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            )
            print(f"Time: {train_end_time}")
            print(f"Batch Duration: {train_start_time - batch_start_time}")
            print(f"Train Duration: {train_end_time - train_start_time}")
            print(f"Elapsed Time: {train_end_time - start_time}")
            print(
                "--------------------------------------------------------------------------------------------"
            )
            batch_start_time = datetime.now().replace(microsecond=0)

    env.close()

    # print total training time
    print(
        "============================================================================================"
    )
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at: ", start_time)
    print("Finished training at: ", end_time)
    print("Total training time: ", end_time - start_time)
    print(
        "============================================================================================"
    )


def entry_point():

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    train_agent(config)


if __name__ == "__main__":
    entry_point()
