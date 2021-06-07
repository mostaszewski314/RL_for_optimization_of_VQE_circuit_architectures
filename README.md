# Reinforcement learning for optimization of variational quantum circuit architectures
This repository contains the base code from the article [Reinforcement learning for optimization of variational quantum circuit architectures]
## Installation

Create conda environment using
```
rl_for_vqe.yml
```

## Experiments
Each experiment is defined in a configuration file
```
configuration_files/
├── global_COBYLA  # LiH - 4-qubit - different bond distances
├── global_Rotosolve  # LiH - 4-qubit - different bond distances
├── local_COBYLA  # LiH - 4-qubit - different bond distances
├── local_Rotosolve  # LiH - 4-qubit - different bond distances
├── lower_bound_energy  # LiH - 4-qubit - lower-bound approximation to the ground-stateenergy 
└── moving_threshold  # LiH  -  6-qubits  -  moving  threshold

```
An example configuration for the experiments that use a lower-bound approximation of the ground-state energy
```
[general]
episodes = 40000

[env]
num_qubits = 4 # Number of qubits 
num_layers = 40 # Max number of actions per episode
fake_min_energy = -10.0604 # lower-bound approximation to the ground-state energy
fn_type = incremental_with_fixed_ends # Type of rewad function (We can enlist here possible alternatives)
accept_err = 4  # Threshold after which the episode ends 
shift_threshold_time = 500 # Number of episodes after which threshold is changed greedily (parameter only for MovingThreshold procedure)
shift_threshold_ball = 0.5e-3 # Amortisation radius (parameter only for MovingThreshold procedure)
success_thresh = 25 # Number of succeses after which amortisation is decreased (parameter only for MovingThreshold procedure)
succ_radius_shift = 10 # The number of times the current amortization radius is to be reduced  (parameter only for MovingThreshold procedure)
succes_switch = 4 # Threshold after which amortisation approach is turend on (parameter only for MovingThreshold procedure)
thresholds = [] (parameter only for VanillaCurriculum procedure)
switch_episodes = [] (parameter only for VanillaCurriculum procedure)
curriculum_type = MovingThreshold #Type of curriculum learning method

[problem] # Definition of quantum chemistry problem
ham_type = LiH
geometry = Li .0 .0 .0; H .0 .0 2.2
taper = 1
mapping = parity

[agent]
batch_size = 1000 # Mini-batch size
memory_size = 20000 # Replay buffer size
neurons = [1000,1000,1000,1000,1000] # Number of neurons per layer
dropout = 0.
learning_rate = 0.0001
angles = 0 # Whether the parameters of the quantum circuit are taken into account in the input of the neural network 
en_state = 1 # Whether the energy corresponding to the current quantum circuit is taken into account in the input of the neural network 
agent_type = DeepQ
agent_class = DQN
init_net = 0 # (Whether to use the pretrained net or not)

update_target_net = 500
final_gamma = 0.005
epsilon_decay = 0.99995
epsilon_min = 0.05
epsilon_restart = 1.0

[non_local_opt]
global_iters = 100 # Number of iteration after each step
method = scipy_each_step # Type of method i.e. local or global optimization
optim_alg = COBYLA # Type of optimization algorithm
local_size = None # If local optimization, then how many angles needs to be optimized (Used for experiments local_COBYLA and local_Rotosolve - set to 5)
```
# Training

Train an agent with the example configuration

```
python3 main.py --seed 1234 --config h_s_0 --experiment_name "lower_bound_energy/" 
```
This will automatically create the following directory 
```
results/
└── lower_bound_energy

```

This folder contains all the saved models and the ```summary_1234.npy``` file, which consists of all the details of the training procedure, along with the obtained circuits and the corresponding energies. 
