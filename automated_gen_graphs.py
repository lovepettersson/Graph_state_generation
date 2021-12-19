from gates import switch, QuditLateSPhoton, QuditEarlySPhoton, X_full_matrix, \
    X_half_matrix
from generate_graphs_and_spin_rot_op import spin_inverse_unitary, gen_new_photon, rot_uni_arb, spin_flip_op, spin_unitary
import numpy as np
import json
from five_line_qubit_tom import gates_state_not_ideal, generate_line_ideal
from tqdm import tqdm
#from four_line_tom import generate_line_ideal


def first_three_photons(numb_photons, gates, parameters, LC_ops):
    """
    :param state: intial state, i.e. spin zero and photons in vacuum
    :param numb_photons: number of photons
    :param gates: gates obtained for "gates_state_not_ideal"
    :param parameters: "kappa and deltaOH"
    :param LC_ops: X-rot on spin and Z-rots on photon 1 and 2
    :return: The four qubit graph which all the rest of the graphs stem from
    """
    X0, Z1, Z2 = LC_ops
    kappa, deltaOH = parameters
    ex, early_gate, late_gate, state, C1, C2, H = gates
    state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    for i in range(numb_photons - 1):
        v, state = switch(state, numb_photons)
    ops = [X0, [Z1, Z2]]
    state = LC(state, ops)
    if numb_photons == 4:
        for i in range(2):
            v, state = switch(state, numb_photons)
        state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
        v, state = switch(state, numb_photons)
        v, state = switch(state, numb_photons)
    else:
        for i in range(numb_photons-1):
            v, state = switch(state, numb_photons)
        state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
        v, state = switch(state, numb_photons)

    return state


def LC(state, ops):
    """
    :param state: current state
    :param ops: list of operations to apply, ops[0] is the X-rot and ops[1] are the Z-rots
    :return: new state after LC-op applied
    """
    state = np.matmul(ops[0], state)
    for i in range(len(ops[1])):
        state = np.matmul(ops[1][i], state)

    return state


def Z_rots(numb_photons, deltaOH):
    """

    :param numb_photons: number of photons
    :param deltaOH: deltaOH.... gates ahve to be sampled every monte step...
    :return: A list of Z-pi-half rots for all qudits, starting with spin.
    """
    Z = []
    square_root = 1 / (2 ** (1 / 2))
    rot_matrix = np.array([[1, 0, 0, 0],
                           [0, (1 + 1j) * square_root, 0, 0],
                           [0, 0, (1 - 1j) * square_root, 0],
                           [0, 0, 0, 1],
                           ])
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex64) # dtype=np.complex128

    Z_rot_spin = np.conjugate(spin_inverse_unitary(0, 0, np.pi / 2, 1 + deltaOH))
    for i in range(numb_photons):
        Z_rot_spin = np.kron(Z_rot_spin, identity)
    Z.append(Z_rot_spin)
    for i in range(numb_photons):
        if i == 0:
            op = np.kron(identity, rot_matrix)
        else:
            op = np.kron(identity, identity)
        for j in range(1, numb_photons):
            if i == j:
                op = np.kron(op, rot_matrix)
            else:
                op = np.kron(op, identity)
        Z.append(op)
    return Z


def X_rots(numb_photons, deltaOH):
    """

    :param numb_photons: number of photons
    :param deltaOH: deltaOH.... gates ahve to be sampled every monte step...
    :return: A list of X-pi-half rots for all qudits, starting with the spin.
    """
    X = []
    square_root = 1 / (2 ** (1 / 2))
    rot_matrix = np.array([[1, 0, 0, 0],
                           [0, 1 * square_root, -1j * square_root, 0],
                           [0, -1j * square_root, 1 * square_root, 0],
                           [0, 0, 0, 1],
                           ])
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex128)

    X_rot_spin = spin_unitary(0, np.pi / 7, 3.5, deltaOH)
    for i in range(numb_photons):
        X_rot_spin = np.kron(X_rot_spin, identity)
    X.append(X_rot_spin)
    for i in range(numb_photons):
        if i == 0:
            op = np.kron(identity, rot_matrix)
        else:
            op = np.kron(identity, identity)
        for j in range(1, numb_photons):
            if i == j:
                op = np.kron(op, rot_matrix)
            else:
                op = np.kron(op, identity)
        X.append(op)

    return X


def extract_LC_sequences(local_ops, Z, X, three_qubit = True):
    """

    :param graph:
    :param local_ops: Should be on the form [[X-rot qubit, [Z-rot qubits], ...]
    :param Z: All Z rots needed for LC-op
    :param X: All X_rots needed for LC-op
    :param three_qubit: If we are calling three_qubit function
    :return: return LC-sequence but with the appropiate operators
    """
    LC_sequence = []
    if three_qubit == True:

        target, neighbours = local_ops
        X_rot = X[target]
        Z_rots = [Z[i] for i in neighbours]
        LC_sequence.append(X_rot)
        for Z in Z_rots:
            LC_sequence.append(Z)
    else:

        target, neighbours = local_ops
        X_rot = X[target]
        Z_rot = [Z[i] for i in neighbours]
        LC_sequence.append(X_rot)
        LC_sequence.append(Z_rot)
    return LC_sequence


def four_qubit_graphs(local_ops, gates, parameters, star = True, numb_photons=3):
    """

    :param state: initial state, i.e. spin in zero and photons in vacuum
    :param local_ops: A list of lists, where the inner list contains an LC-sequence
    :param numb_photons: number of photons is three in the case of four qubits
    :return: final graph state
    """
    X = X_rots(numb_photons, 0)
    Z = Z_rots(numb_photons, 0)
    LC_ops = extract_LC_sequences([0, [1, 2]], Z, X)
    state = first_three_photons(numb_photons, gates, parameters, LC_ops)

    if len(local_ops) == 1:
        local_ops = local_ops[0]
        LC_ops = extract_LC_sequences(local_ops, Z, X, False)
        state = LC(state, LC_ops)
        den_square = np.outer(state, np.conjugate(state).transpose())

        return den_square
    else:

        for LCs in local_ops:
            LCS = extract_LC_sequences(LCs, Z, X, False)
            state = LC(state, LCS)
        den_square = np.outer(state, np.conjugate(state).transpose())
    
        return den_square



def gen_fourth_photon(state, gates, parameters, numb_photons=4):
    kappa, deltaOH = parameters
    ex, early_gate, late_gate, state0, C1, C2, H = gates
    for i in range(numb_photons - 1):
        v, state = switch(state, numb_photons)
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)
    return state



def five_qubit_star(local_ops, gates, parameters, numb_photons=4):
    """

    :param state: initial state, i.e. spin in zero and photons in vacuum
    :param local_ops: A list of lists, where the inner list contains an LC-sequence
    :param numb_photons: number of photons is three in the case of four qubits
    :return: final graph state
    """
    kappa, deltaOH = parameters
    ex, early_gate, late_gate, state, C1, C2, H = gates
    state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)

    return state



def five_qubit_graphs(local_ops, gates, parameters, numb_photons=4):
    """

    :param state: initial state, i.e. spin in zero and photons in vacuum
    :param local_ops: A list of lists, where the inner list contains an LC-sequence
    :param numb_photons: number of photons is three in the case of four qubits
    :return: final graph state
    """
    X = X_rots(numb_photons, 0)
    Z = Z_rots(numb_photons, 0)
    LC_ops = extract_LC_sequences([0, [1, 2]], Z, X)
    state = first_three_photons(numb_photons, gates, parameters, LC_ops)
    lc_4 = local_ops["4"]
    lc_5 = local_ops["5"]


    if len(lc_4) == 0:
        state = gen_fourth_photon(state, gates, parameters)
    elif len(lc_4) == 1:
        lc_4 = lc_4[0]
        LC_ops = extract_LC_sequences(lc_4, Z, X, False)
        state = LC(state, LC_ops)
        state = gen_fourth_photon(state, gates, parameters)
    elif len(lc_4) > 1:
        for LCs in lc_4:
            LCS = extract_LC_sequences(LCs, Z, X, False)
            state = LC(state, LCS)
            state = gen_fourth_photon(state, gates, parameters)


    if len(lc_5) == 0:
        return state
    elif len(lc_5) == 1:
        lc_5 = lc_5[0]
        LC_ops = extract_LC_sequences(lc_5, Z, X, False)
        state = LC(state, LC_ops)
        return state
    else:
        for LCs in lc_5:
            LCS = extract_LC_sequences(LCs, Z, X, False)
            state = LC(state, LCS)
            return state



# DONE:
# X and Z rotations, as well as the appropiate gates

# TODO:
# Extract LC_sequence from dict
# Patch all functions together such that we can generate a given graph with just a number


def get_it(numb, kappa):
    path_catalouge = r"C:\Users\Admin\now_with_lc.json"
    f = open(path_catalouge)
    iter_dict = json.load(f)
    iter_dict = iter_dict[str(numb)]
    local_ops = (iter_dict["lc_ops"])
    print(local_ops)
    gates = gates_state_not_ideal(3, 0.992, 0.072, 0.1)
    deltaOH = np.random.normal(0, np.sqrt(2) / 23.2)
    parameters = [kappa, deltaOH]
    den_matrix = 0
    for i in range(20):
        den_matrix += four_qubit_graphs(local_ops, gates, parameters) / 20
    return den_matrix

def get_five(numb, kappa, T2):
    path_catalouge = r"C:\Users\Admin\now_with_lc_5_qubit.json"
    f = open(path_catalouge)
    iter_dict = json.load(f)
    iter_dict = iter_dict[str(numb)]
    local_ops = (iter_dict["lc_ops"])
    print(local_ops)
    #gates = gates_state_not_ideal(4, 0.992, 0.072, 0.1)
    gates = gates_state_not_ideal(4, 1, 0, 0)
    deltaOH = np.random.normal(0, np.sqrt(2) / T2)
    parameters = [kappa, deltaOH]
    den_matrix = 0
    the_list = [i for i in range(2000)]
    for i in tqdm(the_list):
        #  state = five_qubit_star(local_ops, gates, parameters)
        state = five_qubit_graphs(local_ops, gates, parameters)
        den = np.outer(state, np.conjugate(state).transpose())
        den_matrix += den / 2000
    return den_matrix
