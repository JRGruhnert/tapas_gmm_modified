from dataclasses import dataclass
from datetime import datetime
from omegaconf import OmegaConf, SCMode

from tapas_gmm.master_project.dloader import DataLoader
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
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
    checkpoint: str
    keep_epoch: bool
    verbose: bool = True


def train_agent(config: MasterConfig):
    # Initialize the environment and agent
    dloader = DataLoader(config.state_space, config.task_space, config.verbose)
    env = MasterEnv(config.env, dloader.states, dloader.tasks)
    agent = MasterAgent(
        config.agent,
        config.nt,
        config.tag,
        dloader.states,
        dloader.tasks,
    )
    agent.load(config.checkpoint, config.keep_epoch)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    stop_training = False
    while not stop_training:  # Training loop
        start_time_batch = datetime.now().replace(microsecond=0)
        terminal = False
        batch_rdy = False
        obs, goal = env.reset()
        while not terminal and not batch_rdy:
            task = agent.act(obs, goal)
            reward, terminal, obs = env.step(task, verbose=config.verbose)
            batch_rdy = agent.feedback(reward, terminal)
        if batch_rdy:
            start_time_learning = datetime.now().replace(microsecond=0)
            stop_training = agent.learn(verbose=config.verbose)
            end_time_learning = datetime.now().replace(microsecond=0)
            print(
                f"""
                ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                Batch Duration: {start_time_learning - start_time_batch}
                Learn Duration: {end_time_learning - start_time_learning}
                Elapsed Time:   {end_time_learning - start_time}
                Current Time:   {end_time_learning}
                --------------------------------------------------------------------------------------------
                """
            )
    env.close()
    end_time = datetime.now().replace(microsecond=0)
    print(
        f"""
        ============================================================================================
        Start Time: {start_time}
        End Time:   {end_time}
        Duration:   {end_time - start_time}
        ============================================================================================
        """
    )


def entry_point():

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    train_agent(config)


if __name__ == "__main__":
    entry_point()
