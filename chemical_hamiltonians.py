from qiskit.aqua.operators import Z2Symmetries
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.core import Hamiltonian, QubitMappingType, TransformationType
import numpy as np
from qiskit.aqua.algorithms import VQE, NumPyEigensolver

def translate_pauli_term(pauli_word):
        return ' '.join([pauli_word[i]+' '+str(i) for i in range(len(pauli_word))])

def convert_from_qiskit(hamiltonian):
    """
    Converts given qubit hamiltonian generated in qiskit,
    into list of pauli terms and weights easy to apply in qulacs.
    """
    paulis, weights = list(zip(*[i.split("\t") for i in hamiltonian.split("\n")[:-1]]))

    weights = [complex(i[1:-1]) for i in weights]
    paulis_qulacs = [translate_pauli_term(i)for i in paulis]
    return paulis, paulis_qulacs, weights


def qiskit_LiH_chem(geometry, taper=True, exact=False, mapping='parity'):
    """
    Generates list of pauli terms, weights and shift of the energy, for given 
    molecule of LiH and its Cartesian coordinates of each atomic species.
    Most of this code is taken from Qiskit Textbook
    https://qiskit.org/textbook/ch-applications/vqe-molecules.html
    """

    driver = PySCFDriver(atom=geometry, unit=UnitsType.ANGSTROM, 
                            charge=0, spin=0, basis='sto3g')
    molecule = driver.run()
    freeze_list = [0]
    remove_list = [-3, -2]
    repulsion_energy = molecule.nuclear_repulsion_energy
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    remove_list = [x % molecule.num_orbitals for x in remove_list]
    freeze_list = [x % molecule.num_orbitals for x in freeze_list]
    remove_list = [x - len(freeze_list) for x in remove_list]
    remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]
    freeze_list += [x + molecule.num_orbitals for x in freeze_list]
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)
    ferOp = ferOp.fermion_mode_elimination(remove_list)
    num_spin_orbitals -= len(remove_list)
    qubitOp = ferOp.mapping(map_type=mapping, threshold=0.00000001)
    if taper:
        qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
        
    shift = energy_shift + repulsion_energy
    
    result = NumPyEigensolver(qubitOp,2**qubitOp.num_qubits).run()
    
    if exact:
        exact_energies = np.real(result.eigenvalues) + shift
    else:
        exact_energies = np.real(result.eigenvalues)
    
    paulis, paulis_qulacs, weights = convert_from_qiskit(qubitOp.print_details())
    

    return paulis, paulis_qulacs, weights, exact_energies, shift

def qiskit_H2_chem(geometry, exact=True):
    """
    https://github.com/Qiskit/qiskit-community-tutorials/blob/master/chemistry/h2_vqe_spsa.ipynb
    """
    driver = PySCFDriver(geometry, unit=UnitsType.ANGSTROM, 
                            charge=0, spin=0, basis='sto3g')
    qmolecule = driver.run()
    repulsion_energy = qmolecule.nuclear_repulsion_energy 
    operator =  Hamiltonian(transformation=TransformationType.FULL,
                            qubit_mapping=QubitMappingType.PARITY, two_qubit_reduction=True)
    qubit_op, aux_ops = operator.run(qmolecule)

    paulis, paulis_qulacs, weights = convert_from_qiskit(qubit_op.print_details())

    return paulis, paulis_qulacs, weights, repulsion_energy


def pauliTerm2mtx(pauliTerm):
    Sx = np.array([[0, 1], [1, 0]])
    Sy = np.array([[0, -1j], [1j, 0]])
    Sz = np.array([[1, 0], [0, -1]])
    Sid = np.eye(2)

    operator = []
    for letter in pauliTerm[::-1]:
        if letter == 'I':
            operator.append(Sid)
        elif letter == 'X':
            operator.append(Sx)
        elif letter == 'Y':
            operator.append(Sy)
        elif letter == 'Z':
            operator.append(Sz)
    result = operator[0]
    for j in range(1, len(pauliTerm)):
        result = np.kron(result, operator[j])
    return result.astype('complex64')


def paulis2matrices(paulis):
    return list(map(pauliTerm2mtx, paulis))


if __name__ == "__main__":  
    
    ## Examine functions    
    paulis, paulis_qulacs, weights, energies, shift = qiskit_LiH_chem("Li .0 .0 .0; H .0 .0 2.2",False,True,'jordan_wigner')


    ham = paulis2matrices(paulis)
    tmp = [weights[i]* ham[i] for i in range(len(paulis))]
    hamiltonian = np.sum(tmp, axis=0)
    print(hamiltonian)
    e,u= np.linalg.eig(hamiltonian)
    print(np.real(e.min()))
    print(np.real(e.min())+shift)


    st = np.asarray([ 2.60545687e-05+8.56407639e-09j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j, -1.39737460e-05-4.59313794e-09j,
  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j, -0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j,  1.81323714e-01-5.96006846e-05j,
 -9.72486455e-02+3.19654045e-05j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j, -0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j, -9.60224174e-02-3.15623462e-05j,
  5.14993315e-02+1.69277110e-05j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
 -1.37975482e-05+4.53522213e-09j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j,  7.39998561e-06-2.43235813e-09j,
  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
 -9.72244749e-01-3.19574597e-04j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j,  4.57271386e-04+1.50304045e-07j,
  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j, -1.39702729e-04+4.59199634e-08j,
  6.57057399e-08-2.15973245e-11j, -0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j,  3.31342550e-06+1.08911528e-09j,
 -1.55838812e-09-5.12238566e-13j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
  2.30593959e-02-7.57957002e-06j,  0.00000000e+00+0.00000000e+00j,
  0.00000000e+00+0.00000000e+00j, -1.08454193e-05+3.56486419e-09j])

    target = u[:, np.argmin(e)]

    np.round(target-st, 5)


   
