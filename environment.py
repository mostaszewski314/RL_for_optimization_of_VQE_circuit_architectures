
import torch
from qulacs import QuantumCircuit
from qulacs.gate import CNOT, RX, RY, RZ
from utils import *
from sys import stdout
from itertools import product
import scipy
import VQE as vc
import os
import numpy as np
import random
import copy
import curricula
try:
    from qulacs import QuantumStateGpu as QuantumState
except ImportError:
    from qulacs import QuantumState




class CircuitEnv():

    def __init__(self, conf, device):

        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']

        self.ham_mapping = conf['problem']['mapping']
        self.geometry = conf['problem']['geometry'].replace(" ", "_")

        self.fake_min_energy = conf['env']['fake_min_energy'] if "fake_min_energy" in conf['env'].keys() else None
        self.fn_type = conf['env']['fn_type'] 
   


        # If you want to run agent from scratch without *any* curriculum just use the setting with 
        # normal curriculum and set config[episodes] = [1000000] 
        self.curriculum_dict = {}
        __ham = np.load(f"mol_data/LiH_{self.num_qubits}q_geom_{self.geometry}_{self.ham_mapping}.npz")
        hamiltonian, eigvals, energy_shift = __ham['hamiltonian'], __ham['eigvals'], __ham['energy_shift']

        min_eig = conf['env']['fake_min_energy'] if "fake_min_energy" in conf['env'].keys() else min(eigvals) + energy_shift

        self.curriculum_dict[self.geometry[-3:]] = curricula.__dict__[conf['env']['curriculum_type']](conf['env'], target_energy=min_eig)


        self.device = device
        self.ket = QuantumState(self.num_qubits)
        self.done_threshold = conf['env']['accept_err']
        

        stdout.flush()
        self.state_size = 5*self.num_layers
        self.actual_layer = -1
        self.prev_energy = None
        self.energy = 0

        self.action_size = (self.num_qubits*(self.num_qubits+2))
 

        if 'non_local_opt' in conf.keys():
            self.global_iters = conf['non_local_opt']['global_iters']
            self.optim_method = conf['non_local_opt']["method"]

            if conf['non_local_opt']["method"] in ["Rotosolve_local_end", "Rotosolve_local_end_only_rot", "scipy_local_end"]:
                self.local_opt_size = conf['non_local_opt']["local_size"]
            if "optim_alg" in conf['non_local_opt'].keys():
                self.optim_alg = conf['non_local_opt']["optim_alg"]

        else:
            self.global_iters = 0
            self.optim_method = None


    def step(self, action, train_flag = True) :

        """
        Action is performed on the first empty layer.
        Variable 'actual_layer' points last non-empty layer.
        """
        next_state = self.state.clone()
        self.actual_layer += 1

        """
        First two elements of the 'action' vector describes position of the CNOT gate.
        Position of rotation gate and its axis are described by action[2] and action[3].
        When action[0] == num_qubits, then there is no CNOT gate.
        When action[2] == num_qubits, then there is no Rotation gate.
        """

        next_state[0][self.actual_layer] = action[0]
        next_state[1][self.actual_layer] = (action[0] + action[1]) % self.num_qubits

        ## state[2] corresponds to number of qubit for rotation gate
        next_state[2][self.actual_layer] = action[2]
        next_state[3][self.actual_layer] = action[3]
        next_state[4][self.actual_layer] = torch.zeros(1)

        self.state = next_state.clone()

        # if rotation gate is present, then run the rotosolve
        if next_state[2][self.actual_layer] != self.num_qubits:
            thetas = self.get_angles(self.actual_layer)
            next_state[-1] = thetas

            if self.optim_method == "Rotosolve_local_end_only_rot":
                thetas_to_optim = min(next_state[-1][next_state[2]!=self.num_qubits].size()[0],self.local_opt_size)
                angle_indices = -np.arange(thetas_to_optim, 0, -1)
                thetas = self.global_roto(angle_indices)
                next_state[-1] = thetas

        self.state = next_state.clone()
        if self.optim_method == "Rotosolve_local_end":
            thetas_to_optim = min(next_state[-1][next_state[2]!=self.num_qubits].size()[0],self.local_opt_size)
            angle_indices = -np.arange(thetas_to_optim, 0, -1)
            thetas = self.global_roto(angle_indices)
            next_state[-1] = thetas
        elif self.optim_method in ["scipy_local_end"]:
            nb_of_thetas = next_state[-1][next_state[2]!=self.num_qubits].size()[0] ## number of all thetas
            thetas_to_optim = min(nb_of_thetas, self.local_opt_size) ## number of thetas which we want to optimize
            angle_indices = np.arange(nb_of_thetas - thetas_to_optim, nb_of_thetas) ## in COBYLA case we need them in ascending order and positive (not [-1,-2,-,3])
            # print(angle_indices)
            # print(next_state[-1][next_state[2]!=self.num_qubits])
            if nb_of_thetas != 0:
                thetas = self.scipy_optim(self.optim_alg, angle_indices)
                next_state[-1] = thetas

        elif self.optim_method == "Rotosolve_each_step":
            thetas = self.global_roto()
            next_state[-1] = thetas
        elif self.optim_method in ["scipy_each_step"]:
            if next_state[-1][next_state[2]!=self.num_qubits].size()[0] != 0:
                thetas = self.scipy_optim(self.optim_alg)
                next_state[-1] = thetas

        self.state = next_state.clone()

        energy = self.get_energy()
        self.energy = energy
        if energy < self.curriculum.lowest_energy and train_flag:
            self.curriculum.lowest_energy = copy.copy(energy)
    
        self.error = float(abs(self.min_eig-energy))
     
        rwd = self.reward_fn(energy)
        self.prev_energy = np.copy(energy)

        energy_done = int(self.error < self.done_threshold)
        layers_done = self.actual_layer == (self.num_layers-1)
        done = int(energy_done or layers_done)
     
        if done:
            self.curriculum.update_threshold(energy_done=energy_done)
            self.done_threshold = self.curriculum.get_current_threshold()
            self.curriculum_dict[str(self.current_bond_distance)] = copy.deepcopy(self.curriculum)
        
        return next_state.view(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done

    def reset(self):
        """
        Returns randomly initialized state of environment.
        State is a torch Tensor of size (5 x number of layers)
        1st row [0, num of qubits-1] - denotes qubit with control gate in each layer
        2nd row [0, num of qubits-1] - denotes qubit with not gate in each layer
        3rd, 4th & 5th row - rotation qubit, rotation axis, angle
        !!! When some position in 1st or 3rd row has value 'num_qubits',
            then this means empty slot, gate does not exist (we do not
            append it in circuit creator)
        """
        ## state_per_layer: (Control_qubit, NOT_qubit, R_qubit, R_axis, R_angle)
        controls = self.num_qubits * torch.ones(self.num_layers)
        nots = torch.zeros(self.num_layers)
        rotats = self.num_qubits * torch.ones(self.num_layers)
        generatos = torch.zeros(self.num_layers)
        angles = torch.zeros(self.num_layers)

        state = torch.stack((controls.float(),
                            nots.float(),
                            rotats.float(),
                            generatos.float(),
                            angles))
        self.state = state

        self.make_circuit(state)
        self.actual_layer = -1


        self.current_bond_distance = self.geometry[-3:]
        self.curriculum = copy.deepcopy(self.curriculum_dict[str(self.current_bond_distance)])
        self.done_threshold = copy.deepcopy(self.curriculum.get_current_threshold())

        self.geometry = self.geometry[:-3] + str(self.current_bond_distance)

        __ham = np.load(f"mol_data/LiH_{self.num_qubits}q_geom_{self.geometry}_{self.ham_mapping}.npz")
        self.hamiltonian, eigvals, self.energy_shift = __ham['hamiltonian'], __ham['eigvals'], __ham['energy_shift']

        self.min_eig = self.fake_min_energy if self.fake_min_energy is not None else min(eigvals) + self.energy_shift
        self.max_eig = max(eigvals)+self.energy_shift

        self.prev_energy = self.get_energy(state)

        return state.view(-1).to(self.device)

    def make_circuit(self, thetas=None):
        """
        based on the angle of first rotation gate we decide if any rotation at
        a given qubit is present i.e.
        if thetas[0, i] == 0 then there is no rotation gate on the Control quibt
        if thetas[1, i] == 0 then there is no rotation gate on the NOT quibt
        CNOT gate have priority over rotations when both will be present in the given slot
        """
        state = self.state.clone()
        if thetas is None:
            thetas = state[-1]
        circuit = QuantumCircuit(self.num_qubits)

        for i in range(self.num_layers):
            if state[0][i].item() != self.num_qubits:
                circuit.add_gate(CNOT(int(state[0][i].item()),
                                      int(state[1][i].item())))
            elif state[2][i].item() != self.num_qubits:
                circuit.add_gate(self.R_gate(int(state[2][i].item()),
                                             int(state[3][i].item()),
                                             thetas[i].item()))
        assert circuit.get_gate_count() <= self.num_layers, "Wrong circuit construction, too many gates!!!"
        return circuit

    def R_gate(self, qubit, axis, angle):
        if axis == 'X' or axis == 'x' or axis == 1:
            return RX(qubit, angle)
        elif axis == 'Y' or axis == 'y' or axis == 2:
            return RY(qubit, angle)
        elif axis == 'Z' or axis == 'z' or axis == 3:
            return RZ(qubit, angle)
        else:
            print("Wrong gate")
            return 1

    def get_energy(self, thetas=None):
        circ = self.make_circuit(thetas)
        self.ket.set_zero_state()
        circ.update_quantum_state(self.ket)
        v = self.ket.get_vector()

        return np.real(np.vdot(v,np.dot(self.hamiltonian,v)))+ self.energy_shift
        # return self.observable.get_expectation_value(self.ket) + self.energy_shift

    def get_ket(self, thetas=None):
        state = self.state.clone()
        circ = self.make_circuit(thetas)
        self.ket.set_zero_state()
        circ.update_quantum_state(self.ket)
        v = self.ket.get_vector()
        v_real = torch.tensor(np.real(v),device=self.device,dtype=torch.float)
        v_imag = torch.tensor(np.imag(v),device=self.device,dtype=torch.float)
        return torch.cat((v_real,v_imag)).view(1,-1)

    def get_angles(self, update_idx):
        state = self.state.clone()
        thetas = state[-1]
        thetas[update_idx] = 0.0
        theta1, theta2, theta3 = thetas.clone(), thetas.clone(), thetas.clone()
        theta1[update_idx] -= np.pi/2
        theta3[update_idx] += np.pi/2

        e1 = self.get_energy(theta1)
        e2 = self.get_energy(theta2)
        e3 = self.get_energy(theta3)
        ## Energy lanscape is of the form A \sin( \theta + B) + C
        C = 0.5*(e1+e3)
        B = np.arctan2((e2-C), (e3-C))

        thetas[update_idx] = -B-np.pi/2
        return thetas

    def global_roto(self, which_angles=[]):
        state = self.state.clone()
        thetas = state[-1][state[2]!=self.num_qubits]

        qulacs_inst = vc.Parametric_Circuit(n_qubits=self.num_qubits)
        param_circuit = qulacs_inst.construct_ansatz(state)
        # print("DEPTH",param_circuit.calculate_depth())

        arguments = (self.hamiltonian, param_circuit, self.num_qubits, self.energy_shift)

        if not list(which_angles):
            which_angles = np.arange(len(thetas))

        for j in range(self.global_iters):
            for i in which_angles:

                theta1, theta2, theta3 = thetas.clone(), thetas.clone(), thetas.clone()
                theta1[i] -= np.pi/2
                theta3[i] += np.pi/2

                e1 = vc.get_energy_qulacs(theta1,*arguments)
                e2 = vc.get_energy_qulacs(theta2,*arguments)
                e3 = vc.get_energy_qulacs(theta3,*arguments)
                ## Energy lanscape is of the form A \sin( \theta + B) + C
                C = 0.5*(e1+e3)
                B = np.arctan2((e2-C), (e3-C)) - theta2[i]

                thetas[i] = -B-np.pi/2

        state[-1][state[2]!=self.num_qubits] = thetas
        return state[-1]

    def scipy_optim(self, method, which_angles=[]):
        state = self.state.clone()
        thetas = state[-1][state[2]!=self.num_qubits]

        qulacs_inst = vc.Parametric_Circuit(n_qubits=self.num_qubits)
        qulacs_circuit = qulacs_inst.construct_ansatz(state)

        x0 = np.asarray(thetas.cpu().detach())
        if list(which_angles):
            # print(which_angles)
            # print(x0)
            result_min_qulacs = scipy.optimize.minimize(vc.get_energy_qulacs, x0=x0[which_angles],
                                                            args=(self.hamiltonian,
                                                                  qulacs_circuit,
                                                                  self.num_qubits,
                                                                  self.energy_shift,
                                                                  which_angles),
                                                            method=method,
                                                            options={'maxiter':self.global_iters})
            # print(result_min_qulacs)
            x0[which_angles] = result_min_qulacs['x']
            state[-1][state[2]!=self.num_qubits] = torch.tensor(x0, dtype=torch.float)
        else:
            result_min_qulacs = scipy.optimize.minimize(vc.get_energy_qulacs, x0=x0,
                                                        args=(self.hamiltonian,
                                                              qulacs_circuit,
                                                              self.num_qubits,
                                                              self.energy_shift),
                                                        method=method,
                                                        options={'maxiter':self.global_iters})
            state[-1][state[2]!=self.num_qubits] = torch.tensor(result_min_qulacs['x'], dtype=torch.float)

        return  state[-1]


    def reward_fn(self, energy):
        if self.fn_type == "staircase":
            return (0.2 * (self.error < 15 * self.done_threshold) +
                    0.4 * (self.error < 10 * self.done_threshold) +
                    0.6 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold)) / 2.2
        elif self.fn_type == "two_step":
            return (0.001 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold))/1.001
        elif self.fn_type == "two_step_end":
            max_depth = self.actual_layer == (self.num_layers - 1)
            if ((self.error < self.done_threshold) or max_depth):
                return (0.001 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold))/1.001
            else:
                return 0.0
        elif self.fn_type == "naive":
            return 0. + 1.*(self.error < self.done_threshold)
        elif self.fn_type == "incremental":
            return (self.prev_energy - energy)/abs(self.prev_energy - self.min_eig)
        elif self.fn_type == "incremental_clipped":
            return np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
        elif self.fn_type == "nive_fives":
            max_depth = self.actual_layer == (self.num_layers-1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = 0.
            return rwd
        elif self.fn_type == "incremental_with_fixed_ends":
            max_depth = self.actual_layer == (self.num_layers-1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
            return rwd
        elif self.fn_type == "log":
            return -np.log(1-(energy/self.min_eig))
        elif self.fn_type == "log_neg_punish":
            return -np.log(1-(energy/self.min_eig)) - 5
        elif self.fn_type == "end_energy":
            max_depth = self.actual_layer == (self.num_layers - 1)
            if ((self.error < self.done_threshold) or max_depth):
                rwd = (self.max_eig - energy) / (abs(self.min_eig) + abs(self.max_eig))
            else:
                rwd = 0.0
            return rwd





if __name__ == "__main__":
    pass