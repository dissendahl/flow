"""Runner script for single and multi-agent reinforcement learning experiments.

This script performs an RL experiment using the DDPG algorithm. Choice of
hyperparameters can be seen and adjusted from the code below.

Usage
    python train.py EXP_CONFIG
"""

import argparse
import json
import os
import sys
from time import strftime
from flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

import ray
from ray import tune
from ray.tune import run_experiments, grid_search
from ray.tune.registry import register_env, register_trainable
from flow.utils.registry import make_create_env

from ray.rllib.contrib.maddpg.maddpg import MADDPGTrainer, DEFAULT_CONFIG

from copy import deepcopy

from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params

NUM_CPUS = 1
NUM_ITERATIONS = 250

class CustomStdOut(object):
    def _log_result(self, result):
        if result["training_iteration"] % 50 == 0:
            try:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    result["timesteps_total"],
                    result["episodes_total"],
                    result["episode_reward_mean"],
                    result["policy_reward_mean"],
                    round(result["time_total_s"] - self.cur_time, 3)
                ))
            except:
                pass

            self.cur_time = result["time_total_s"]


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent or exp_configs/rl/multiagent.')

    # optional input parameters
    parser.add_argument(
        '--rl_trainer', type=str, default="RLlib",
        help='the RL trainer to use. either RLlib or Stable-Baselines')

    parser.add_argument(
        '--num_cpus', type=int, default=1,
        help='How many CPUs to use')
    parser.add_argument(
        '--num_steps', type=int, default=5000,
        help='How many total steps to perform learning over')
    parser.add_argument(
        '--rollout_size', type=int, default=1000,
        help='How many steps are in a training batch.')

    return parser.parse_known_args(args)[0]


def run_model_stablebaseline(flow_params, num_cpus=1, rollout_size=50, num_steps=50):
    """Run the model for num_steps if provided.

    Parameters
    ----------
    num_cpus : int
        number of CPUs used during training
    rollout_size : int
        length of a single rollout
    num_steps : int
        total number of training steps
    The total rollout length is rollout_size.

    Returns
    -------
    stable_baselines.*
        the trained model
    """
    if num_cpus == 1:
        constructor = env_constructor(params=flow_params, version=0)()
        # The algorithms require a vectorized environment to run
        env = DummyVecEnv([lambda: constructor])
    else:
        env = SubprocVecEnv([env_constructor(params=flow_params, version=i)
                             for i in range(num_cpus)])

    train_model = PPO2('MlpPolicy', env, verbose=1, n_steps=rollout_size)
    train_model.learn(total_timesteps=num_steps)
    return train_model


def setup_exps_rllib(flow_params,
                     n_cpus,
                     n_rollouts,
                     n_iterations,
                     policy_graphs=None,
                     policy_mapping_fn=None,
                     policies_to_train=None):
    """Return the relevant components of an RLlib experiment.

    Parameters
    ----------
    flow_params : dict
        flow-specific parameters (see flow/utils/registry.py)
    n_cpus : int
        number of CPUs to run the experiment over
    n_rollouts : int
        number of rollouts per training iteration
    policy_graphs : dict, optional
        TODO
    policy_mapping_fn : function, optional
        TODO
    policies_to_train : list of str, optional
        TODO

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """

    # config params
    alg_run = "contrib/MADDPG"

    # config params
    horizon = flow_params['env'].horizon

    # config params
    config = deepcopy(DEFAULT_CONFIG)
    config["num_workers"] = n_cpus
    config["train_batch_size"] = horizon * n_rollouts
    config["learning_starts"] = config["train_batch_size"] * n_iterations * 0.1
    config["horizon"] = horizon
    config["tau"] = grid_search([3e-3, 2e-3, 1e-3, 5e-4, 1e-4])
    config["critic_lr"] = grid_search([1e-2, 1e-3, 1e-4])
    config["actor_lr"] = grid_search([1e-2, 1e-3, 1e-4])
    config["grad_norm_clipping"] = None
    config["actor_feature_reg"] = None
    config["log_level"] = "INFO"

    ##ToDo: Inspect if and which values to set for these two hyperparameters
    config["ignore_worker_failures"] = True
    config["use_local_critic"] = True


        # === Exploration ===
    exploration_config = {
        # DDPG uses OrnsteinUhlenbeck (stateful) noise to be added to NN-output
        # actions (after a possible pure random phase of n timesteps).
        "type": "OrnsteinUhlenbeckNoise",
        # For how many timesteps should we return completely random actions,
        # before we start adding (scaled) noise?
        "random_timesteps": 10000,
        # The OU-base scaling factor to always apply to action-added noise.
        "ou_base_scale": 1,
        # The OU theta param.
        "ou_theta": 0.15,
        # The OU sigma param.
        "ou_sigma": 0.2,
        # The initial noise scaling factor.
        "initial_scale": 1.0,
        # The final noise scaling factor.
        "final_scale": 1.0,
        # Timesteps over which to anneal scale (from initial to final values).
        "scale_timesteps": 10000,
    }

    config["exploration_config"] = exploration_config

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    config['multiagent'].update({'policies': policy_graphs})
    config['multiagent'].update({'policy_mapping_fn': policy_mapping_fn})
    config['multiagent'].update({'policies_to_train': policies_to_train})

    create_env, gym_name = make_create_env(params=flow_params)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


if __name__ == "__main__":
    flags = parse_args(sys.argv[1:])

    # import relevant information from the exp_config script
    module = __import__("exp_configs.rl.singleagent", fromlist=[flags.exp_config])
    module_ma = __import__("exp_configs.rl.multiagent", fromlist=[flags.exp_config])
    if hasattr(module, flags.exp_config):
        submodule = getattr(module, flags.exp_config)
    elif hasattr(module_ma, flags.exp_config):
        submodule = getattr(module_ma, flags.exp_config)
        assert flags.rl_trainer == "RLlib", \
            "Currently, multiagent experiments are only supported through "\
            "RLlib. Try running this experiment using RLlib: 'python train.py EXP_CONFIG'"
    else:
        assert False, "Unable to find experiment config!"

        ## THIS ONE HERE
    if flags.rl_trainer == "RLlib":
        flow_params = submodule.flow_params
        n_cpus = submodule.N_CPUS
        n_rollouts = submodule.N_ROLLOUTS
        n_iterations = submodule.N_ITERATIONS


        # Imported from multiagent_ppo.py
        policy_graphs = getattr(submodule, "POLICY_GRAPHS", None)
        policy_mapping_fn = getattr(submodule, "policy_mapping_fn", None)
        policies_to_train = getattr(submodule, "policies_to_train", None)

        alg_run, gym_name, config = setup_exps_rllib(
            flow_params, n_cpus, n_rollouts, n_iterations,
            policy_graphs, policy_mapping_fn, policies_to_train)

        ray.init(num_cpus=n_cpus + 1)
        trials = run_experiments({
            flow_params["exp_tag"]: {
                "run": alg_run,
                "env": gym_name,
                "config": {
                    **config
                },
                "checkpoint_freq": 25,
                "checkpoint_at_end": True,
                "max_failures": 2,
                "stop": {
                    "training_iteration": n_iterations,
                },
            }
        })

        print(trials)
    else:
        assert False, "rl_trainer should be either 'RLlib' or 'Stable-Baselines'!"
