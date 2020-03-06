#!/bin/bash
conda activate maddpg
cd examples
export SUMO_HOME="$HOME/sumo_binaries/bin"
export PATH="$SUMO_HOME:$PATH"

# Running experiment with global critic, multiple policies
echo "Starting DDPG training"
python3.6 train_ddpg.py multiagent_ddpg_multi &> examples/results/ddpg_multi_policy.log &
echo "Finished DDPG training"

# Deleting version of ray installed & installing 0.9.0dev with exploration API.
echo "Reconfiguring Ray"
pip uninstall ray -y
pip install https://ray-wheels.s3-us-west-2.amazonaws.com/master/2d97650b1e01c299eda8d973c3b7792b3ac85307/ray-0.9.0.dev0-cp36-cp36m-macosx_10_13_intel.whl

#Running MADDPG with hyperparameter optimisation.
echo "Starting MADDPG training"
python3.6 train_maddpg_hyper.py multiagent_maddpg &> examples/results/maddpg_hyper.log &
echo "Finished MADDPG training"
conda deactivate maddpg
