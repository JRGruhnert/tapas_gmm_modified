from dataclasses import dataclass
import itertools
import numpy as np
from omegaconf import OmegaConf, SCMode
import concurrent.futures

from conf.master.shared.experiment import Exp1Config, ExperimentConfig
from tapas_gmm.master_project.agent import MasterAgent
from tapas_gmm.master_project.dloader import DataLoader
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from tapas_gmm.master_project.environment import MasterEnv
from tapas_gmm.utils.argparse import parse_and_build_config


@dataclass
class EvalConfig:
    state_space: StateSpace
    task_space: TaskSpace
    tag: str
    experiment: ExperimentConfig
    checkpoint: str = ""


def eval_agent(config: EvalConfig):
    # Initialize the environment and agent
    dloader = DataLoader(
        config.state_space, config.task_space, config.experiment.verbose
    )
    env = MasterEnv(config.experiment.env, dloader.states, dloader.tasks)
    agent = MasterAgent(
        config.experiment.agent,
        config.experiment.nt,
        config.tag,
        dloader.states,
        dloader.tasks,
    )
    agent.load(config.checkpoint)

    # track total training time
    stop_evaluating = False
    while not stop_evaluating:  # Training loop
        terminal = False
        obs, goal = env.reset()
        while not terminal:
            task = agent.act(obs, goal, eval=True)
            reward, terminal, obs = env.step_exp1(
                task, verbose=config.experiment.verbose
            )
            stop_evaluating = agent.feedback(reward, terminal)
    print(f"Eval Done for: {config.tag}")
    env.close()
    del dloader
    del env
    del agent  # Free up memory


def entry_point():

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    config: Exp1Config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.workers) as executor:
        # This will submit tasks as workers become free

        # If negative make all
        pe_values = (
            np.arange(
                config.p_min, config.p_max + config.p_step, config.p_step
            ).tolist()
            if config.pe < 0
            else [config.pe]
        )
        pr_values = (
            np.arange(
                config.p_min, config.p_max + config.p_step, config.p_step
            ).tolist()
            if config.pr < 0
            else [config.pr]
        )
        jobs = list(itertools.product(pe_values, pr_values))
        total_jobs = len(jobs)
        for pe, pr in jobs:
            if pe > 0.0 and pr > 0.0:
                pe = pe / 2
                pr = pr / 2
            print(f"Starting evaluations for pe={pe}, pr={pr}")
            for origin, goal in config.cross_t:
                suffix = f"_pe_{pe}_pr_{pr}/model_cp_best.pth"
                prefix = f"results/{config.nt.value}/t"
                eval_config = EvalConfig(
                    state_space=config.t_spaces_mapping[origin]["state_space"],
                    task_space=config.t_spaces_mapping[goal]["task_space"],
                    checkpoint=f"{prefix}{origin}{goal}{suffix}",
                    tag=f"e{origin}{goal}",
                    experiment=config,
                )
                executor.submit(eval_agent, eval_config)
            for i, future in enumerate(
                concurrent.futures.as_completed(executor._pending_work_items), 1
            ):
                print(f"\rProgress: {i}/{total_jobs} Last: pe={pe}, pr={pr}", end="")
        executor.shutdown(wait=True)  # Wait for all tasks to finish
        print("All evaluations done.")


if __name__ == "__main__":
    entry_point()
