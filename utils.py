
import configparser
import numpy as np 
from chemical_hamiltonians import qiskit_LiH_chem, convert_from_qiskit, qiskit_H2_chem, paulis2matrices
import json
from itertools import product
from qulacs import Observable



def gen_hamiltonian(num_qubits, conf, taper=True, exact_en=False):
    if conf["ham_type"] == 'LiH':        
        paulis, paulis_qulacs, weights, energies, shift = qiskit_LiH_chem(conf["geometry"], conf["taper"], exact_en, conf["mapping"])
      
        # observable = Observable(num_qubits)
        # for i in range(len(paulis)):
        #     observable.add_operator(weights[i], paulis_qulacs[i])
        ham = paulis2matrices(paulis)
        tmp = [weights[i]* ham[i] for i in range(len(paulis))]
        hamiltonian = np.sum(tmp, axis=0)
        
        return hamiltonian, energies, shift
    elif conf["ham_type"] == 'H2':
        assert num_qubits == 2, "H2 molecule has 2 qubit Hamiltonian"
        paulis, paulis_qulacs, w, shift = qiskit_H2_chem(conf["geometry"])   
        # observable = Observable(num_qubits)    
        # for i in range(len(paulis)):
        #     observable.add_operator(weights[i], paulis_qulacs[i])
        ham = paulis2matrices(paulis)
        tmp = [w[i]* ham[i] for i in range(len(paulis))]
        hamiltonian = np.sum(tmp, axis=0)
        eigvals, eigvecs = np.linalg.eig(hamiltonian)
        return hamiltonian, eigvals.real, shift



def get_config(config_name,experiment_name, path='configuration_files',
               verbose=True):
    config_dict = {}
    Config = configparser.ConfigParser()
    Config.read('{}/{}{}'.format(path,config_name,experiment_name))
    for sections in Config:
        config_dict[sections] = {}
        for key, val in Config.items(sections):
            # config_dict[sections].update({key: json.loads(val)})
            try:
                config_dict[sections].update({key: int(val)})
            except ValueError:
                config_dict[sections].update({key: val})
            floats = ['learning_rate',  'dropout', 'alpha', 
                      'beta', 'beta_incr', 
                      "shift_threshold_ball","succes_switch","tolearance_to_thresh","memory_reset_threshold",
                      "fake_min_energy","_true_en"]
            strings = ['ham_type', 'fn_type', 'geometry','method','agent_type',
                       "agent_class","init_seed","init_path","init_thresh","method",
                       "mapping","optim_alg", "curriculum_type"]
            lists = ['episodes','neurons', 'accept_err','epsilon_decay',"epsilon_min",
                     "epsilon_decay",'final_gamma','memory_clean',
                     'update_target_net', 'epsilon_restart', "thresholds", "switch_episodes"]
            if key in floats:
                config_dict[sections].update({key: float(val)})
            elif key in strings:
                config_dict[sections].update({key: str(val)})
            elif key in lists:
                config_dict[sections].update({key: json.loads(val)})
    del config_dict['DEFAULT']
    return config_dict


def dictionary_of_actions(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations.
    """
    dictionary = dict()
    i = 0
         
    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] =  [c, x, num_qubits, 0]
        i += 1
   
    """h  denotes rotation axis. 1, 2, 3 -->  X, Y, Z axes """
    for r, h in product(range(num_qubits),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary
        
def dict_of_actions_revert_q(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations. Systems have reverted order to above dictionary of actions.
    """
    dictionary = dict()
    i = 0
         
    for c, x in product(range(num_qubits-1,-1,-1),
                        range(num_qubits-1,0,-1)):
        dictionary[i] =  [c, x, num_qubits, 0]
        i += 1
   
    """h  denotes rotation axis. 1, 2, 3 -->  X, Y, Z axes """
    for r, h in product(range(num_qubits-1,-1,-1),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary
        

if __name__ == '__main__':
    from dataclasses import dataclass
    
    
    @dataclass
    class Config:
        num_qubits = 4
        problem={"ham_type" : 'LiH',
        "geometry" : 'Li .0 .0 .0; H .0 .0 2.2',
        "taper" : 1,
        "mapping" : 'parity'}

        
    __ham = dict()    
 
    __ham['hamiltonian'], __ham['eigvals'], __ham['energy_shift'] = gen_hamiltonian(Config.num_qubits, Config.problem)
    __geometry = Config.problem['geometry'].replace(" ", "_")
    np.savez(f"mol_data/LiH_{Config.num_qubits}q_geom_{__geometry}_{Config.problem['mapping']}",**__ham)
   
