from dataclasses import dataclass
from datetime import datetime
import os
import numpy as np
from omegaconf import OmegaConf, SCMode

from tapas_gmm.master_project.buffer import RolloutBuffer
from tapas_gmm.master_project.dloader import DataLoader
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from tapas_gmm.master_project.environment import MasterEnv, MasterEnvConfig
from tapas_gmm.master_project.agent import MasterAgent, AgentConfig
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.utils.argparse import parse_and_build_config


@dataclass
class EvalConfig:
    state_space: StateSpace
    task_space: TaskSpace
    tag: str
    env: MasterEnvConfig
    verbose: bool = True
    num_samples: int = 500


def eval_task(config: EvalConfig):
    # Initialize the environment and agent
    dloader = DataLoader(config.state_space, config.task_space, config.verbose)
    env = MasterEnv(config.env, dloader.states, dloader.tasks)
    buffer = RolloutBuffer()
    os.makedirs("results/tasks", exist_ok=True)
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    task_stats: dict["str", dict[str, float]] = {}
    for task in dloader.tasks:
        start_time_batch = datetime.now().replace(microsecond=0)
        task_stats[task.name] = {"reward": [], "terminal": []}
        for _ in range(config.num_samples):
            obs, goal = env.reset(task)
            reward, terminal, obs = env.step(task, verbose=config.verbose)
            task_stats[task.name]["reward"].append(reward)
            task_stats[task.name]["terminal"].append(terminal)
        numpy_stats = {k: np.array(v) for k, v in task_stats[task.name].items()}
        np.savez(f"results/tasks/stats_{task.name}.npz", **numpy_stats)
        end_time_batch = datetime.now().replace(microsecond=0)
        print(
            f"""
            Task: {task.name}
            Batch Duration: {end_time_batch - start_time_batch}
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
    print("Saving Task Stats...")
    for task_name, stats in task_stats.items():
        print(f"Task: {task_name}, Stats: {stats}")


def entry_point():

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    eval_task(config)


if __name__ == "__main__":
    entry_point()
