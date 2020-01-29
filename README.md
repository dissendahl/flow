# Repo Fork to demonstrate problem of MADDPG algorithm applied to traffic light grid environmet.

Hi there. This folk exists to demonstrate how the MADDPG implementation in rllib/contrib results in no learning when applied to a specific environment. To give some context: The flow project links traffic simulators with Rllib to setup RL based experiments to investigate with autonomous vehicles or traffic lights (More to see below in the original description). 

I want to apply MADDPG to an environment called traffic-light-grid in which RL traffic lights learn to route the traffic efficiently. They share the reward, the cumulative delay of all cars throughout the simulation steps in a fully cooperative manner. When I employ PPO or DDPG both algorithms learn how to route the traffic quite efficiently, while MADDPG does not. 
[Here](https://github.com/dissendahl/flow/tree/master/learning_curves) you can see the learning curves.

So something with the experiment configuration or the contrib/MADDPG implementation must be off. 
Here are the experiment configs and how to execute them:

* [My training config for MADDPG -> this one does not learn at all](https://github.com/dissendahl/flow/blob/master/examples/exp_configs/rl/multiagent/multiagent_maddpg.py)
* Command to start MADDPG training (executed from within /examples dir): python train_maddpg.py multiagent_maddpg
* [My training configuration for DDPG for comparison -> this one learns](https://github.com/dissendahl/flow/blob/master/examples/exp_configs/rl/multiagent/multiagent_ddpg.py)
* Command to start DDPG training (executed within /examples dir): python train_ddpg.py multiagent_ddpg

To understand, what happens within the experiment configuration and especially its lower part (policy configuration), I refer to the accompanying [tutorial notebook](https://github.com/dissendahl/flow/blob/master/tutorials/tutorial14_mutiagent.ipynb)


### I appreciate every piece of information & help.



------------------------------------------------------------------------------------------------------------------------------

<img src="docs/img/square_logo.png" align="right" width="25%"/>

[![Build Status](https://travis-ci.com/flow-project/flow.svg?branch=master)](https://travis-ci.com/flow-project/flow)
[![Docs](https://readthedocs.org/projects/flow/badge)](http://flow.readthedocs.org/en/latest/)
[![Coverage Status](https://coveralls.io/repos/github/flow-project/flow/badge.svg?branch=master)](https://coveralls.io/github/flow-project/flow?branch=master)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flow-project/flow/binder)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/flow-project/flow/blob/master/LICENSE.md)

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
