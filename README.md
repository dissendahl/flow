### Installation & setup
##### Create environment & setup flow
```shell
conda create --name maddpg python=3.6 -y
conda activate maddpg
python3.6 setup.py develop
```

##### Install sumo (traffic simulator) binaries
```shell
scripts/setup_sumo_osx.sh
#scripts/setup_sumo_ubuntu1804.sh
export SUMO_HOME="$HOME/sumo_binaries/bin"
export PATH="$SUMO_HOME:$PATH"
```

##### Install specific Ray version with refactored exploration noise API
```shell
conda create --name maddpg python=3.6 -y
conda activate maddpg
python3.6 setup.py develop
```

### Experiment documentation
1. Non-RL actuated lights baseline - Can be run by:
```shell
python simulate.py traffic_light_grid_edit
```
See [rendered simulation](examples/results/renderings/baseline.mov) and [simulation metrics](examples/results/simulation_metrics/baseline.txt).

2. PPO - No artefacts stored.
![See](examples/results/screen_shots/ppo.png)

3. TD3 - Converged to policy where all traffic lights toggle synchronously.
See [rendered simulation](examples/results/renderings/td3.mov) and [simulation metrics](examples/results/simulation_metrics/td3_225.txt).
![See](examples/results/screen_shots/td3.png)

4. DDPG - One actor per traffic light / Mutual critic. Training terminated due to leaking experience buffer. See [rendered simulation](examples/results/renderings/ddpg_multi_policy.mov) and [simulation metrics](examples/results/simulation_metrics/ddpg_multi_agent_25.txt). When observing the simulation, we can see that this setup - employing local actor / mutual critic - leads to individual actions which we could argue demonstrates, that the setup picks up local optimisation signals while learning.
![See](examples/results/screen_shots/ddpg_with_local_policies.png)

5. MADDPG - No learning. Possible due to missing noise.
![See](examples/results/screen_shots/maddpg.png)

6. MADDPG - Copied noise parameters from DDPG config

### Repo Documentation for Error Inspection - Fork to demonstrate problem of MADDPG algorithm applied to traffic light grid environmet.

Hi there. This folk exists to demonstrate how the MADDPG implementation in rllib/contrib results in no learning when applied to a specific environment. To give some context: The flow project links traffic simulators with Rllib to setup RL based experiments to investigate with autonomous vehicles or traffic lights (More to see below in the original description).

I want to apply MADDPG to an environment called traffic-light-grid in which RL traffic lights learn to route the traffic efficiently. They share the reward, the cumulative delay of all cars throughout the simulation steps in a fully cooperative manner. When I employ PPO or DDPG both algorithms learn how to route the traffic in one way or another, while MADDPG does not show any learning.
[Here](https://github.com/dissendahl/flow/tree/master/learning_curves) you can see the learning curves.

So something with the experiment configuration or the contrib/MADDPG implementation must be off.
Here are the experiment configs and how to execute them:

* [Training config for MADDPG -> this one does not learn at all](https://github.com/dissendahl/flow/blob/master/examples/exp_configs/rl/multiagent/multiagent_maddpg.py)
* Command to start MADDPG training (executed from within /examples dir):
```shell
python train_maddpg.py multiagent_maddpg
```
* [Training configuration for DDPG for comparison -> this one learns](https://github.com/dissendahl/flow/blob/master/examples/exp_configs/rl/multiagent/multiagent_ddpg.py)
* Command to start DDPG training (executed within /examples dir):
```shell
python train_ddpg.py multiagent_ddpg
```

To understand, what happens within the experiment configuration and especially its lower part (policy configuration), I refer to the accompanying [tutorial notebook](https://github.com/dissendahl/flow/blob/master/tutorials/tutorial14_mutiagent.ipynb)

------------------------------------------------------------------------------------------------------------------------------
# Flow

[Flow](https://flow-project.github.io/) is a computational framework for deep RL and control experiments for traffic microsimulation.

See [our website](https://flow-project.github.io/) for more information on the application of Flow to several mixed-autonomy traffic scenarios. Other [results and videos](https://sites.google.com/view/ieee-tro-flow/home) are available as well.

# More information

- [Documentation](https://flow.readthedocs.org/en/latest/)
- [Installation instructions](http://flow.readthedocs.io/en/latest/flow_setup.html)
- [Tutorials](https://github.com/flow-project/flow/tree/master/tutorials)
- [Binder Build (beta)](https://mybinder.org/v2/gh/flow-project/flow/binder)

# Technical questions

If you have a bug, please report it. Otherwise, join the [Flow Users group](https://forms.gle/CuVBu6QtX3dfNaxz6) on Slack! You'll recieve an email shortly after filling out the form.

# Getting involved

We welcome your contributions.

- Please report bugs and improvements by submitting [GitHub issue](https://github.com/flow-project/flow/issues).
- Submit your contributions using [pull requests](https://github.com/flow-project/flow/pulls). Please use [this template](https://github.com/flow-project/flow/blob/master/.github/PULL_REQUEST_TEMPLATE.md) for your pull requests.

# Citing Flow

If you use Flow for academic research, you are highly encouraged to cite our paper:

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol. abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465

If you use the benchmarks, you are highly encouraged to cite our paper:

Vinitsky, E., Kreidieh, A., Le Flem, L., Kheterpal, N., Jang, K., Wu, F., ... & Bayen, A. M,  Benchmarks for reinforcement learning in mixed-autonomy traffic. In Conference on Robot Learning (pp. 399-409). Available: http://proceedings.mlr.press/v87/vinitsky18a.html

# Contributors

Flow is supported by the [Mobile Sensing Lab](http://bayen.eecs.berkeley.edu/) at UC Berkeley and Amazon AWS Machine Learning research grants. The contributors are listed in [Flow Team Page](https://flow-project.github.io/team.html).
