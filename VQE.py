

from qulacs import ParametricQuantumCircuit, QuantumState, Observable, QuantumCircuit
from qulacs.observable import create_observable_from_openfermion_text
from qulacs.gate import CNOT, RX, RY, RZ

import os

import scipy
import time
import matplotlib.pyplot as plt
import torch
import pathlib

import numpy as np
from sys import argv, stdout

import argparse
from mlflow import log_metric, log_param, set_experiment

from utils import get_config, dictionary_of_actions, gen_hamiltonian

import cma


# -----------------------------------------------------------------------------

class Parametric_Circuit:
    def __init__(self,n_qubits):
        self.n_qubits = n_qubits
        self.ansatz = ParametricQuantumCircuit(n_qubits)

    def construct_ansatz(self, state):
        thetas = state[-1]
        for i in range(state.shape[1]):
            if state[0][i].item() != self.n_qubits:
                self.ansatz.add_CNOT_gate(int(state[0][i].item()),
                                        int(state[1][i].item()))
            elif state[2][i].item() != self.n_qubits:
                axis = int(state[3][i].item())
                if axis =='X' or axis == 'x' or axis == 1:
                    self.ansatz.add_parametric_RX_gate(int(state[2][i].item()), thetas[i])
                elif axis =='Y' or axis == 'y' or axis== 2:
                    self.ansatz.add_parametric_RY_gate(int(state[2][i].item()), thetas[i])
                elif axis =='Z' or axis == 'z' or axis == 3:
                    self.ansatz.add_parametric_RZ_gate(int(state[2][i].item()), thetas[i])
        assert self.ansatz.get_gate_count() <= state.shape[1], "Wrong circuit construction, too many gates!!!"
        return self.ansatz
    

def get_energy_qulacs(angles, observable, circuit, n_qubits, energy_shift, which_angles=[]):
    """"
    Function for Qiskit energy minimization using Qulacs
    
    Input:
    angles                [array]      : list of trial angles for ansatz
    observable            [Observable] : Qulacs observable (Hamiltonian)
    circuit               [circuit]    : ansatz circuit
    n_qubits              [int]        : number of qubits
    energy_shift          [float]      : energy shift for Qiskit Hamiltonian after freezing+removing orbitals
    
    Output:
    expval [float] : expectation value 
    
    """
        
    parameter_count_qulacs = circuit.get_parameter_count()
    param_qulacs = [circuit.get_parameter(ind) for ind in range(parameter_count_qulacs)]    
    if not list(which_angles):
            which_angles = np.arange(parameter_count_qulacs)
    
    for i, j in enumerate(which_angles):
        circuit.set_parameter(j, angles[i])
        
    state = QuantumState(n_qubits)
    circuit.update_quantum_state(state)   
    v = state.get_vector()
    return np.real(np.vdot(v,np.dot(observable,v))) + energy_shift





if __name__ == "__main__":
    pass


















