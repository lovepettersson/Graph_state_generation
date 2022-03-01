import numpy as np


def spin_unitary(phi, rabi, time, deltaOH):
    norm = 1 / np.sqrt((rabi) ** 2 + (deltaOH ** 2))
    theta = (time / 2) / norm
    identity_part = np.array([[1 * np.cos(theta), 0, 0, 0],
                              [0, 1 * np.cos(theta), 0, 0],
                              [0, 0, 1 / 4, 0],
                              [0, 0, 0, 1 / 4]])

    x_part = np.array([[0, 1 * (-1j) * np.sin(theta) * rabi*np.cos(phi) * norm, 0, 0],
                       [1 * (-1j) * np.sin(theta) * rabi*np.cos(phi) * norm, 0, 0, 0],
                       [0, 0, 1 / 4, 0],
                       [0, 0, 0, 1 / 4]])

    y_part = np.array([[0, -1j * (-1j) * np.sin(theta) * rabi*np.sin(phi) * norm, 0, 0],
                       [1j * (-1j) * np.sin(theta) * rabi*np.sin(phi) * norm, 0, 0, 0],
                       [0, 0, 1 / 4, 0],
                       [0, 0, 0, 1 / 4]])

    z_part = np.array([[1 * (-1j) * np.sin(theta) * (-deltaOH) * norm, 0, 0, 0],
                       [0, -1 * (-1j) * np.sin(theta) * (-deltaOH) * norm, 0, 0],
                       [0, 0, 1 / 4, 0],
                       [0, 0, 0, 1 / 4]]
                                  )

    rotation_unitary = identity_part + y_part + x_part + z_part
    return rotation_unitary


def spin_inverse_unitary(phi, rabi, time, deltaOH):
    norm = 1 / np.sqrt((rabi) ** 2 + (deltaOH ** 2))
    theta = (time / 2) / norm
    identity_part = np.array([[1 * np.cos(theta), 0, 0, 0],
                              [0, 1 * np.cos(theta), 0, 0],
                              [0, 0, 1 / 4, 0],
                              [0, 0, 0, 1 / 4]])

    x_part = np.array([[0, 1 * (1j) * np.sin(theta) * rabi*np.cos(phi) * norm, 0, 0],
                       [1 * (1j) * np.sin(theta) * rabi*np.cos(phi) * norm, 0, 0, 0],
                       [0, 0, 1 / 4, 0],
                       [0, 0, 0, 1 / 4]])

    y_part = np.array([[0, -1j * (1j) * np.sin(theta) * rabi*np.sin(phi) * norm, 0, 0],
                       [1j * (1j) * np.sin(theta) * rabi*np.sin(phi) * norm, 0, 0, 0],
                       [0, 0, 1 / 4, 0],
                       [0, 0, 0, 1 / 4]])

    z_part = np.array([[1 * (1j) * np.sin(theta) * (-deltaOH) * norm, 0, 0, 0],
                       [0, -1 * (1j) * np.sin(theta) * (-deltaOH) * norm, 0, 0],
                       [0, 0, 1 / 4, 0],
                       [0, 0, 0, 1 / 4]]
                                  )

    rotation_unitary = identity_part + y_part + x_part + z_part
    return rotation_unitary




def switch(new_state_switch, numb_sites):
    """ Basis change when switching photon 3 with photon 1, or any other photon with 1
     used later in switch density"""
    new_state = new_state_switch
    new_state1 = []
    new_state0 = []
    splitter = 4 ** numb_sites
    length = int(len(new_state_switch) / splitter)

    #### instead of 16 and 4 use 4**numb of sites and (4**numb of sites ) / 4
    for i in range(length):
        flip0 = new_state[i * splitter:i * splitter + splitter]
        flip = np.array(np.arange(i * splitter, i * splitter + splitter, 1))
        four_by_four = []
        four_by_four0 = []
        for j in range(4):
            four_by_four.append([flip[j * int(splitter/4): j * int(splitter/4) + int(splitter/4)]])
            four_by_four0.append([flip0[j * int(splitter/4): j * int(splitter/4) + int(splitter/4)]])
        four_by_four0 = np.asarray(four_by_four0)
        four_by_four0 = four_by_four0.transpose()
        four_by_four = np.asarray(four_by_four)
        four_by_four = four_by_four.transpose()
        flipped0 = np.reshape(four_by_four0, splitter)
        flipped = np.reshape(four_by_four, splitter)
        flipped = list(flipped)
        flipped0 = list(flipped0)
        for id in flipped:
            new_state1.append(id)
        new_state0.append(flipped0)
    new_state1 = np.asarray(new_state1)
    empty = switch_state(new_state_switch, new_state1)
    return new_state1, empty


def switch_state(state, v):
    new_state = []
    for i in range(len(state)):
        new_state.append(state[v[i]])
    new_state = np.asarray(new_state)
    return new_state


def LC(state, ops, include_X=False):
    """
    :param state: current state
    :param ops: list of operations to apply, ops[0] is the X-rot and ops[1] are the Z-rots
    :return: new state after LC-op applied
    """
    if include_X == False:
        # state = np.matmul(ops[0], state)
        for i in range(len(ops[1])):
            state = np.matmul(ops[1][i], state)
    else:
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


def measurement_rot_unitary(theta, axis, phi):
    identity_part = np.array([[1 / 4, 0, 0, 0],
                              [0, 1 * np.cos(theta), 0, 0],
                              [0, 0, 1 * np.cos(theta), 0],
                              [0, 0, 0, 1 / 4]])

    x_part = np.array([[1 / 4, 0, 0, 0],
                       [0, 0, 1 * (-1j) * np.sin(theta) * np.cos(phi), 0],
                       [0, 1 * (-1j) * np.sin(theta) * np.cos(phi), 0, 0],
                       [0, 0, 0, 1 / 4]])

    y_part = np.array([[1 / 4, 0, 0, 0],
                       [0, 0, -1j * (-1j) * np.sin(theta) * np.sin(phi), 0],
                       [0, 1j * (-1j) * np.sin(theta) * np.sin(phi), 0, 0],
                       [0, 0, 0, 1 / 4]])

    z_part = np.array([[1 / 4, 0, 0, 0],
                       [0, 1 * (-1j) * np.sin(theta), 0, 0],
                       [0, 0, -1 * (-1j) * np.sin(theta), 0],
                       [0, 0, 0, 1 / 4]]
                      )
    if axis == "Z":
        rotation_unitary = identity_part + z_part

    elif axis == "X":
        rotation_unitary = identity_part + x_part

    elif axis == "Y":
        rotation_unitary = identity_part + y_part

    else:
        rotation_unitary = identity_part + y_part + x_part + z_part

    return rotation_unitary


def prob_dist(kappa):
    t_f = np.random.exponential(1/kappa)
    return t_f


def spin_flip_op(kappa):
    zero_to_one = np.array([[0, 0, 0, 0],
                            [np.sqrt(kappa), 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])

    one_to_zero = np.array([[0, np.sqrt(kappa), 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])

    return zero_to_one, one_to_zero



def rot_uni_arb(t_dur, state_evolved, kappa, numb_photons, C1, C2, deltaOH, psi_s):
    t_now = 0
    psi = state_evolved
    iden = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    while t_now < t_dur:
        t_f = prob_dist(kappa)
        if t_now + t_f >= t_dur:
            pulse = spin_unitary(psi_s, np.pi / 7, t_dur - t_now, deltaOH) # Change 0 to pi/ 2 for y-rot
            rot = pulse
            for i in range(numb_photons):
                rot = np.kron(rot, iden)
            psi = np.matmul(rot, psi)
            break
        else:
            pulse = spin_unitary(psi_s, np.pi / 7, t_f, deltaOH) # Change 0 to pi/ 2 for y-rot
            rot = pulse
            for i in range(numb_photons):
                rot = np.kron(rot, iden)
            psi = np.matmul(rot, psi)
            p1 = np.linalg.norm(np.matmul(C1, psi)) ** 2
            p2 = np.linalg.norm(np.matmul(C2, psi)) ** 2
            ### EXACTLY LIKE MARTIN
            if np.random.uniform() < p1:
                psi = np.matmul(C1, psi) / (np.sqrt(p1))
            else:
                psi = np.matmul(C2, psi) / (np.sqrt((p2)))

            t_now += t_f
    return psi


def rot_uni_inv(t_dur, state_evolved, kappa, numb_photons, C1, C2, deltaOH):
    t_now = 0
    psi = state_evolved
    iden = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    while t_now < t_dur:
        t_f = prob_dist(kappa)
        if t_now + t_f >= t_dur:
            pulse = spin_inverse_unitary(0, np.pi / 7, t_dur - t_now, deltaOH) # Change 0 to pi/ 2 for y-rot
            rot = pulse
            for i in range(numb_photons):
                rot = np.kron(rot, iden)
            psi = np.matmul(rot, psi)
            break
        else:
            pulse = spin_inverse_unitary(0, np.pi / 7, t_f, deltaOH) # Change 0 to pi/ 2 for y-rot
            rot = pulse
            for i in range(numb_photons):
                rot = np.kron(rot, iden)
            psi = np.matmul(rot, psi)
            p1 = np.linalg.norm(np.matmul(C1, psi)) ** 2
            if np.random.uniform() < p1:
                psi = np.matmul(C1, psi) / (np.sqrt(p1))
            else:
                psi = np.matmul(C2, psi) / (np.sqrt((1 - p1)))

            t_now += t_f
    return psi


def optical_collapse_operators(early_late, I, cyclicity):
    eta = 1
    c_perp = np.sqrt((eta) * cyclicity / (cyclicity + 1))

    photon_early = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 0]])

    photon_late = np.array([[0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 1, 0]])


    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex64)

    C1_one = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])

    C1_two = np.array([[0, 0, 0, 0],
                   [0, 0, np.sqrt(I)*c_perp, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])


    C2 = np.array([[0, 0, 0, 0],
                   [0, 0, np.sqrt((1-eta)*cyclicity/(cyclicity+1)), 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])

    C2 = np.kron(C2, identity)

    C3 = np.array([[0, 0, 0, 0],
                   [0, 0, c_perp * np.sqrt(1 - I), 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])


    C4 = np.array([[0, 0, 0, np.sqrt(cyclicity / (cyclicity + 1))],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])

    C4 = np.kron(C4, identity)

    C5 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, np.sqrt(1 / (cyclicity + 1))],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])

    C5 = np.kron(C5, identity)

    C6 = np.array([[0, 0, np.sqrt(1 / (cyclicity + 1)), 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])

    C6 = np.kron(C6, identity)
    if early_late == "early":
        C1_one = np.kron(C1_one, identity)
        C1_two = np.kron(C1_two, photon_early)
        C1 = C1_one + C1_two
        C3 = np.kron(C3, photon_early)
    else:
        C1_one = np.kron(C1_one, identity)
        C1_two = np.kron(C1_two, photon_late)
        C1 = C1_one + C1_two
        C3 = np.kron(C3, photon_late)


    C = [C1, C2, C3, C4, C5, C6]

    return C


def excite_collapse_operators():

    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex64)

    omega00 = 0.93498729+0.34202655j
    omega30 = -0.00313928+0.00259632j
    omega11 = 0.025488
    omega21 = -0.94305002j
    omega11p_lose = 0.08
    omega21p_lose = 0.12
    omega00p = 0.08767089380158649
    omega10p = 0.16
    omega01p = 0.021286938180070818
    omega02p_lose = 0.021139211501800235
    omega03p_lose = 0.02671730487962699
    omega13p_lose = 0.03312317579068492


    C1 = np.array([[omega00, 0, 0, 0],
                  [0, omega11, 0, 0],
                  [0, omega21, 0, 0],
                  [omega30, 0, 0, 0]])

    C4 = np.array([[0, 0, 0, 0],
                  [0, omega11p_lose, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

    C5 = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, omega21p_lose, 0, 0],
                  [0, 0, 0, 0]])

    C6 = np.array([[omega00p, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

    C7 = np.array([[0, omega01p, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

    C8 = np.array([[0, 0, 0, 0],
                  [omega10p, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

    C9 = np.array([[0, 0, omega02p_lose, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])

    C10 = np.array([[0, 0, 0, omega03p_lose],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])

    C11 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, omega13p_lose],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])

    C = []
    C_prime = [C1, C4, C5, C6, C7, C8, C9, C10, C11]
    for c in C_prime:
        c_full = np.kron(c, identity)
        C.append(c_full)

    return C