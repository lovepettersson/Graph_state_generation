import numpy as np
import random
from generate_graphs_and_spin_rot_op import spin_unitary, spin_flip_op, rot_uni_arb, switch, spin_inverse_unitary, rot_uni_inv
from tqdm import tqdm
from Utility import LC, X_rots, Z_rots, extract_LC_sequences





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


def get_full_ops(numb_photons, ops):
    new_ops = []
    for op in ops:
        for i in range(numb_photons - 1):
            op = np.kron(op, identity)
        new_ops.append(op)
    return new_ops



def decay_op(psi, early_late, I, cyclicity, numb_photons, flag=True):
    C = optical_collapse_operators(early_late, I, cyclicity)
    C = get_full_ops(numb_photons, C)
    P = []
    for c in C:
        psi_ket = np.matmul(c, psi)
        psi_bra = np.conjugate(psi_ket).transpose()
        p = np.matmul(psi_bra, psi_ket)
        P.append(p)
    if flag == True:
        choice = random.choices(range(len(P)), weights=P)[0]
    else:
        choice = 0
    collapse = C[choice]
    prob = P[choice]
    psi = np.matmul(collapse, psi) / np.sqrt(prob)
    return psi


def excite_op(psi, numb_photons, flag=True):
    C = excite_collapse_operators()
    C = get_full_ops(numb_photons, C)
    P = []
    for c in C:
        psi_ket = np.matmul(c, psi)
        psi_bra = np.conjugate(psi_ket).transpose()
        p = np.matmul(psi_bra, psi_ket)
        P.append(p)
    if flag == True:
        choice = random.choices(range(len(P)),  weights=P)[0]
    else:
        choice = 0
    collapse = C[choice]
    prob = P[choice]
    psi = np.matmul(collapse, psi) / np.sqrt(prob)
    return psi


def post_select_op(numb_photons):
    early = np.array([0, 0, 1, 0])
    late = np.array([0, 1, 0, 0])
    post_select = np.kron(identity, (np.outer(early, early.transpose()) + np.outer(late, late.transpose())))
    # post_select = np.kron(identity, no_photons)
    for i in range(numb_photons - 1):
        # post_select = np.kron(post_select, no_photons)
        post_select = np.kron(post_select, (np.outer(early, early.transpose()) + np.outer(late, late.transpose())))
    return post_select


identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex64)



def ghz(I, kappa, T2):
    psi_sim = 0
    C = 14.7
    numb_photons = 1
    C1, C2 = spin_flip_op(kappa)
    # C1 = np.kron(np.kron(C1, identity), identity)
    # C2 = np.kron(np.kron(C2, identity), identity)
    C1 = np.kron(C1, identity)
    C2 = np.kron(C2, identity)
    the_list = [i for i in range(1000)]
    for i in tqdm(the_list):
        # deltaOH = np.random.normal(0, np.sqrt(2) / T2)
        deltaOH = 0
        spin = np.array([1, 0, 0, 0])
        photon = np.array([1, 0, 0, 0])
        psi = np.kron(spin, photon)
        psi = rot_uni_arb(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
        psi = excite_op(psi, numb_photons)
        psi = decay_op(psi, "early", I, C, numb_photons)
        psi = rot_uni_arb(7, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
        psi = excite_op(psi, numb_photons)
        psi = decay_op(psi, "late", I, C, numb_photons)
        psi = rot_uni_arb(7, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
        # v, psi = switch(psi, numb_photons)
        # psi = excite_op(psi, numb_photons)
        # psi = decay_op(psi, "early", I, C, numb_photons)
        # psi = rot_uni_arb(7, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
        # psi = excite_op(psi, numb_photons)
        # psi = decay_op(psi, "late", I, C, numb_photons)
        # psi = rot_uni_arb(7, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
        # v, psi = switch(psi, numb_photons)
        if np.isnan(psi[0]):
            continue
        else:
            den = np.outer(psi, np.conjugate(psi).transpose())
            psi_sim += den / len(the_list)
    post_select = post_select_op(numb_photons)
    psi_sim = np.matmul(np.matmul(post_select, psi_sim), post_select)
    norm = np.trace(psi_sim)
    psi_sim = psi_sim / norm
    return psi_sim

def switch_density(density_matrix, new_v_state, new_u_state, size):
    """ switching order of photon 3 and photon 1, or and other photon with one"""
    new_density_matrix = np.zeros((size, size), dtype=np.complex128)
    for row in range(len(density_matrix[0])):
        for column in range(len(density_matrix[0])):
            new_density_matrix[row][column] = density_matrix[new_u_state[row]][new_v_state[column]]
    return new_density_matrix


def gen_deutsch_state(psi, kappa, C1, C2, I, C, deltaOH, numb_photons, flag=True):
    # Initialization
    psi = rot_uni_inv(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
    # psi = gen_photon(psi, kappa, C1, C2, I, C, deltaOH, numb_photons, flag=True)
    # psi = gen_photon(psi, kappa, C1, C2, I, C, deltaOH, numb_photons, flag=True)

    # First qubit
    psi = excite_op(psi, numb_photons, flag)
    psi = decay_op(psi, "early", I, C, numb_photons, flag)
    psi = rot_uni_arb(7, psi, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
    psi = excite_op(psi, numb_photons, flag)
    psi = decay_op(psi, "late", I, C, numb_photons, flag)
    psi = rot_uni_arb(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
    v, psi = switch(psi, numb_photons)
    # Second qubit
    psi = excite_op(psi, numb_photons, flag)
    psi = decay_op(psi, "early", I, C, numb_photons, flag)
    psi = rot_uni_arb(7, psi, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
    psi = excite_op(psi, numb_photons, flag)
    psi = decay_op(psi, "late", I, C, numb_photons, flag)
    psi = rot_uni_inv(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
    v, psi = switch(psi, numb_photons)
    return psi


def gen_photon(psi, kappa, C1, C2, I, C, deltaOH, numb_photons, flag=True):
    psi = excite_op(psi, numb_photons, flag)
    psi = decay_op(psi, "early", I, C, numb_photons, flag)
    psi = rot_uni_arb(7, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
    psi = excite_op(psi, numb_photons, flag)
    psi = decay_op(psi, "late", I, C, numb_photons, flag)
    psi = rot_uni_arb(7, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
    # psi = np.matmul(H, psi)
    v, psi = switch(psi, numb_photons)
    return psi

def gen_photon_VQE(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, angle_shift):
    psi = excite_op(psi, numb_photons)
    psi = decay_op(psi, "early", I, C, numb_photons)
    psi = rot_uni_arb(7, psi, 0.0000000000001, numb_photons, C1, C2, 0, 0)
    psi = excite_op(psi, numb_photons)
    psi = decay_op(psi, "late", I, C, numb_photons)
    psi = rot_uni_arb(7 + angle_shift, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
    # psi = np.matmul(H, psi)
    # v, psi = switch(psi, numb_photons)
    return psi



def gen_photon_VQE_two(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, angle_shift):
    psi = excite_op(psi, numb_photons)
    psi = decay_op(psi, "early", I, C, numb_photons)
    psi = rot_uni_arb(7, psi, kappa, numb_photons, C1, C2, 0, 0)
    psi = excite_op(psi, numb_photons)
    psi = decay_op(psi, "late", I, C, numb_photons)
    psi = rot_uni_arb(7 + angle_shift, psi, kappa, numb_photons, C1, C2, deltaOH, angle_shift)
    return psi

def gen_con_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, flag=True):
    psi = excite_op(psi, numb_photons, flag)
    psi = decay_op(psi, "early", I, C, numb_photons, flag)
    psi = rot_uni_arb(7, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
    psi = excite_op(psi, numb_photons, flag)
    psi = decay_op(psi, "late", I, C, numb_photons,flag)
    psi = rot_uni_arb(7, psi, 0.00000000000000001, numb_photons, C1, C2, deltaOH, 0)
    psi = np.matmul(H, psi)
    v, psi = switch(psi, numb_photons)
    return psi


def gen_line(I, kappa, T2, numb_photons):
    psi_sim = 0
    C = 14000000000000000000.7
    C1, C2 = spin_flip_op(kappa)
    H_matrix = np.array([[1, 0, 0, 0],
                         [0, 1 / (2 ** (1 / 2)), 1 / (2 ** (1 / 2)), 0],
                         [0, 1 / (2 ** (1 / 2)), -1 / (2 ** (1 / 2)), 0],
                         [0, 0, 0, 1],
                         ])
    spin = np.array([1, 0, 0, 0])
    photon = np.array([1, 0, 0, 0])
    psi_init = np.kron(spin, photon)
    H = np.kron(identity, H_matrix)
    for i in range(numb_photons):
        if i == 0:
            C2 = np.kron(C2, identity)
            C1 = np.kron(C1, identity)
        else:
            H = np.kron(H, identity)
            C1 = np.kron(C1, identity)
            C2 = np.kron(C2, identity)
            psi_init = np.kron(psi_init, photon)
    X = X_rots(numb_photons, 0)
    Z = Z_rots(numb_photons, 0)
    the_list = [i for i in range(1000)]
    for i in range(1000):
        deltaOH = np.random.normal(0, np.sqrt(2) / T2)
        # deltaOH = 0
        psi = psi_init
        psi = rot_uni_arb(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
        psi = gen_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons)
        psi = gen_con_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons)

        ###### LC NUMBER ONE ######
        v, psi = switch(psi, numb_photons)
        psi = rot_uni_arb(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
        LC_ops = extract_LC_sequences([0, [1, 2]], Z, X)
        X0, Z1, Z2 = LC_ops
        ops = [X0, [Z1, Z2]]
        psi = LC(psi, ops)
        v, psi = switch(psi, numb_photons)
        v, psi = switch(psi, numb_photons)

        ######## LC 2 ########
        psi = gen_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons)

        ####### LC 3 ######
        LC_ops = extract_LC_sequences([1, [0, 2]], Z, X)
        X1, Z0, Z2 = LC_ops
        ops = [X1, [Z0, Z2]]
        psi = LC(psi, ops, True)

        den = np.outer(psi, np.conjugate(psi).transpose())
        psi_sim += den / len(the_list)

    post_select = post_select_op(numb_photons)
    psi_sim = np.matmul(np.matmul(post_select, psi_sim), post_select)
    norm = np.trace(psi_sim)
    psi_sim = psi_sim / norm
    return psi_sim



def gen_line_ideal(I, kappa, T2, numb_photons):
    psi_sim = 0
    C = 14999999999999.7
    C1, C2 = spin_flip_op(kappa)
    H_matrix = np.array([[1, 0, 0, 0],
                         [0, 1 / (2 ** (1 / 2)), 1 / (2 ** (1 / 2)), 0],
                         [0, 1 / (2 ** (1 / 2)), -1 / (2 ** (1 / 2)), 0],
                         [0, 0, 0, 1],
                         ])
    spin = np.array([1, 0, 0, 0])
    photon = np.array([1, 0, 0, 0])
    psi_init = np.kron(spin, photon)
    H = np.kron(identity, H_matrix)
    for i in range(numb_photons):
        if i == 0:
            C2 = np.kron(C2, identity)
            C1 = np.kron(C1, identity)
        else:
            H = np.kron(H, identity)
            C1 = np.kron(C1, identity)
            C2 = np.kron(C2, identity)
            psi_init = np.kron(psi_init, photon)
    X = X_rots(numb_photons, 0)
    Z = Z_rots(numb_photons, 0)
    the_list = [i for i in range(1)]
    for i in tqdm(the_list):
        deltaOH = np.random.normal(0, np.sqrt(2) / T2)
        # deltaOH = 0
        psi = psi_init
        psi = rot_uni_arb(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
        psi = gen_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, flag = False)
        psi = gen_con_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, flag=False)

        ###### LC NUMBER ONE ######
        v, psi = switch(psi, numb_photons)
        psi = rot_uni_arb(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
        LC_ops = extract_LC_sequences([0, [1, 2]], Z, X)
        X0, Z1, Z2 = LC_ops
        ops = [X0, [Z1, Z2]]
        psi = LC(psi, ops)
        v, psi = switch(psi, numb_photons)
        v, psi = switch(psi, numb_photons)

        ######## LC 2 ########
        psi = gen_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, flag=False)

        ####### LC 3 ######
        LC_ops = extract_LC_sequences([1, [0, 2]], Z, X)
        X1, Z0, Z2 = LC_ops
        ops = [X1, [Z0, Z2]]
        psi = LC(psi, ops, True)

    return psi


def generate_deutsch_state(I, kappa, T2, numb_photons):
    psi_sim = 0
    C = 14.7
    C1, C2 = spin_flip_op(kappa)
    spin = np.array([1, 0, 0, 0])
    photon = np.array([1, 0, 0, 0])
    psi_init = np.kron(spin, photon)
    # psi_init = spin
    for i in range(numb_photons):
        if i == 0:
            C2 = np.kron(C2, identity)
            C1 = np.kron(C1, identity)
        else:
            C1 = np.kron(C1, identity)
            C2 = np.kron(C2, identity)
            psi_init = np.kron(psi_init, photon)

    the_list = [i for i in range(10000)]
    for i in tqdm(the_list):
        deltaOH = np.random.normal(0, np.sqrt(2) / T2)
        psi = gen_deutsch_state(psi_init, kappa, C1, C2, I, C, deltaOH, numb_photons)
        den = np.outer(psi, np.conjugate(psi).transpose())
        psi_sim += den / len(the_list)

    post_select = post_select_op(numb_photons)
    psi_sim = np.matmul(np.matmul(post_select, psi_sim), post_select)
    norm = np.trace(psi_sim)
    psi_sim = psi_sim / norm
    return psi_sim


def gen_box_ideal(I, kappa, T2, numb_photons):
    psi_sim = 0
    C = 1400000000000
    C1, C2 = spin_flip_op(kappa)
    H_matrix = np.array([[1, 0, 0, 0],
                         [0, 1 / (2 ** (1 / 2)), 1 / (2 ** (1 / 2)), 0],
                         [0, 1 / (2 ** (1 / 2)), -1 / (2 ** (1 / 2)), 0],
                         [0, 0, 0, 1],
                         ])
    spin = np.array([1, 0, 0, 0])
    photon = np.array([1, 0, 0, 0])
    psi_init = np.kron(spin, photon)
    H = np.kron(identity, H_matrix)
    for i in range(numb_photons):
        if i == 0:
            C2 = np.kron(C2, identity)
            C1 = np.kron(C1, identity)
        else:
            H = np.kron(H, identity)
            C1 = np.kron(C1, identity)
            C2 = np.kron(C2, identity)
            psi_init = np.kron(psi_init, photon)
    X = X_rots(numb_photons, 0)
    Z = Z_rots(numb_photons, 0)
    the_list = [i for i in range(1)]
    for i in tqdm(the_list):
        deltaOH = 0
        psi = psi_init
        psi = rot_uni_arb(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
        psi = gen_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, flag=False)
        psi = gen_con_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, flag = False)

        ###### LC NUMBER ONE ######
        v, psi = switch(psi, numb_photons)
        psi = rot_uni_arb(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
        LC_ops = extract_LC_sequences([0, [1, 2]], Z, X)
        X0, Z1, Z2 = LC_ops
        ops = [X0, [Z1, Z2]]
        psi = LC(psi, ops)
        v, psi = switch(psi, numb_photons)
        v, psi = switch(psi, numb_photons)

        ######## LC 2 ########
        psi = gen_con_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, flag = False)
        psi = rot_uni_arb(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
        LC_ops = extract_LC_sequences([0, [1, 2, 3]], Z, X)
        X0, Z1, Z2, Z3 = LC_ops
        ops = [X0, [Z1, Z2, Z3]]
        psi = LC(psi, ops)

        ####### LC 3 ######
        LC_ops = extract_LC_sequences([1, [0, 3]], Z, X)
        X1, Z0, Z3 = LC_ops
        ops = [X1, [Z0, Z3]]
        psi = LC(psi, ops, True)

    return psi



def gen_box(I, kappa, T2, numb_photons):
    psi_sim = 0
    C = 14.7
    C1, C2 = spin_flip_op(kappa)
    H_matrix = np.array([[1, 0, 0, 0],
                         [0, 1 / (2 ** (1 / 2)), 1 / (2 ** (1 / 2)), 0],
                         [0, 1 / (2 ** (1 / 2)), -1 / (2 ** (1 / 2)), 0],
                         [0, 0, 0, 1],
                         ])
    spin = np.array([1, 0, 0, 0])
    photon = np.array([1, 0, 0, 0])
    psi_init = np.kron(spin, photon)
    H = np.kron(identity, H_matrix)
    for i in range(numb_photons):
        if i == 0:
            C2 = np.kron(C2, identity)
            C1 = np.kron(C1, identity)
        else:
            H = np.kron(H, identity)
            C1 = np.kron(C1, identity)
            C2 = np.kron(C2, identity)
            psi_init = np.kron(psi_init, photon)
    X = X_rots(numb_photons, 0)
    Z = Z_rots(numb_photons, 0)
    the_list = [i for i in range(1000)]
    for i in tqdm(the_list):
        deltaOH = np.random.normal(0, np.sqrt(2) / T2)
        # deltaOH = 0
        psi = psi_init
        psi = rot_uni_arb(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
        psi = gen_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons)
        psi = gen_con_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons)

        ###### LC NUMBER ONE ######
        v, psi = switch(psi, numb_photons)
        psi = rot_uni_arb(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
        LC_ops = extract_LC_sequences([0, [1, 2]], Z, X)
        X0, Z1, Z2 = LC_ops
        ops = [X0, [Z1, Z2]]
        psi = LC(psi, ops)
        v, psi = switch(psi, numb_photons)
        v, psi = switch(psi, numb_photons)

        ######## LC 2 ########
        psi = gen_con_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons)
        psi = rot_uni_arb(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
        LC_ops = extract_LC_sequences([0, [1, 2, 3]], Z, X)
        X0, Z1, Z2, Z3 = LC_ops
        ops = [X0, [Z1, Z2, Z3]]
        psi = LC(psi, ops)

        ####### LC 3 ######
        LC_ops = extract_LC_sequences([1, [0, 3]], Z, X)
        X1, Z0, Z3 = LC_ops
        ops = [X1, [Z0, Z3]]
        psi = LC(psi, ops, True)

        den = np.outer(psi, np.conjugate(psi).transpose())
        psi_sim += den / len(the_list)

    post_select = post_select_op(numb_photons)
    psi_sim = np.matmul(np.matmul(post_select, psi_sim), post_select)
    norm = np.trace(psi_sim)
    psi_sim = psi_sim / norm
    return psi_sim
    # psi_sim = psi_sim / norm
    # return psi_sim




def three_star_VQE_two(I, numb_photons, angle_shift0, angle_shift1, kappa):
    C = 1444444444444444444444.7
    # kappa = 0.021
    C1, C2 = spin_flip_op(kappa)
    H_matrix = np.array([[1, 0, 0, 0],
                         [0, 1 / (2 ** (1 / 2)), 1 / (2 ** (1 / 2)), 0],
                         [0, 1 / (2 ** (1 / 2)), -1 / (2 ** (1 / 2)), 0],
                         [0, 0, 0, 1],
                         ])
    spin = np.array([1, 0, 0, 0])
    photon = np.array([1, 0, 0, 0])
    psi_init = np.kron(spin, photon)
    H = np.kron(identity, H_matrix)
    psi_sim = 0
    for i in range(numb_photons):
        if i == 0:
            C2 = np.kron(C2, identity)
            C1 = np.kron(C1, identity)
        else:
            H = np.kron(H, identity)
            C1 = np.kron(C1, identity)
            C2 = np.kron(C2, identity)
            psi_init = np.kron(psi_init, photon)

    deltaOH = 0
    # deltaOH = np.random.normal(0, np.sqrt(2) / 23.2)
    # for i in range(100):
    #     psi = psi_init
    #     psi = rot_uni_arb(3.5 + angle_shift0, psi, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
    #     psi = gen_photon_VQE_two(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, angle_shift1)
    #     den = np.outer(psi, np.conjugate(psi).transpose())
    #     psi_sim += den / 100
    psi = gen_photon_VQE(psi_init, kappa, C1, C2, I, C, deltaOH, H, numb_photons, angle_shift0)
    psi = gen_photon_VQE(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, angle_shift1)
    psi_sim = np.outer(psi, np.conjugate(psi).transpose())
    post_select = post_select_op(numb_photons)
    psi_sim = np.matmul(np.matmul(post_select, psi_sim), post_select)
    norm = np.trace(psi_sim)
    psi_sim = psi_sim / norm

    # return psi
    return psi_sim



def three_star_VQE(I, numb_photons, angle_shift0, angle_shift1, kappa, T2):
    C = 60
    # kappa = 0.021
    C1, C2 = spin_flip_op(kappa)
    H_matrix = np.array([[1, 0, 0, 0],
                         [0, 1 / (2 ** (1 / 2)), 1 / (2 ** (1 / 2)), 0],
                         [0, 1 / (2 ** (1 / 2)), -1 / (2 ** (1 / 2)), 0],
                         [0, 0, 0, 1],
                         ])
    spin = np.array([1, 0, 0, 0])
    photon = np.array([1, 0, 0, 0])
    psi_init = np.kron(spin, photon)
    H = np.kron(identity, H_matrix)
    psi_sim = 0
    for i in range(numb_photons):
        if i == 0:
            C2 = np.kron(C2, identity)
            C1 = np.kron(C1, identity)
        else:
            H = np.kron(H, identity)
            C1 = np.kron(C1, identity)
            C2 = np.kron(C2, identity)
            psi_init = np.kron(psi_init, photon)

    # deltaOH = 0
    deltaOH = np.random.normal(0, np.sqrt(2) / T2)
    numb_samples = 100
    for i in range(numb_samples):
        psi = psi_init
        psi = rot_uni_arb(3.5 + angle_shift0, psi, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
        psi = gen_photon_VQE(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, angle_shift1)
        den = np.outer(psi, np.conjugate(psi).transpose())
        psi_sim += den / numb_samples
    # psi = gen_photon_VQE(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, angle_shift2)
    # psi = gen_photon_VQE(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, angle_shift3)
    post_select = post_select_op(numb_photons)
    psi_sim = np.matmul(np.matmul(post_select, psi_sim), post_select)
    norm = np.trace(psi_sim)
    psi_sim = psi_sim / norm

    # return psi
    return psi_sim


def three_star(I, kappa, T2, numb_photons, graph):
    psi_sim = 0
    C = 14444444444.7
    C1, C2 = spin_flip_op(kappa)
    H_matrix = np.array([[1, 0, 0, 0],
                         [0, 1 / (2 ** (1 / 2)), 1 / (2 ** (1 / 2)), 0],
                         [0, 1 / (2 ** (1 / 2)), -1 / (2 ** (1 / 2)), 0],
                         [0, 0, 0, 1],
                         ])
    spin = np.array([1, 0, 0, 0])
    photon = np.array([1, 0, 0, 0])
    psi_init = np.kron(spin, photon)
    H = np.kron(identity, H_matrix)
    for i in range(numb_photons):
        if i == 0:
            C2 = np.kron(C2, identity)
            C1 = np.kron(C1, identity)
        else:
            H = np.kron(H, identity)
            C1 = np.kron(C1, identity)
            C2 = np.kron(C2, identity)
            psi_init = np.kron(psi_init, photon)

    the_list = [i for i in range(1000)]
    for i in tqdm(the_list):
        deltaOH = np.random.normal(0, np.sqrt(2) / T2)
        # deltaOH = 0
        psi = psi_init
        psi = rot_uni_arb(3.5, psi, 0.00000000000001, numb_photons, C1, C2, deltaOH, np.pi / 2)
        for j in range(numb_photons - 1):
            psi = gen_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons)

        if graph == "star-middle":
            psi = gen_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons)

        elif graph == "fully":
            psi = gen_con_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons)
            X = X_rots(numb_photons, 0)
            Z = Z_rots(numb_photons, 0)
            psi = rot_uni_arb(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
            LC_ops = extract_LC_sequences([0, [1, 2, 3]], Z, X)
            X0, Z1, Z2, Z3 = LC_ops
            ops = [X0, [Z1, Z2, Z3]]
            psi = LC(psi, ops)

        else:
            psi = gen_con_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons)
            X = X_rots(numb_photons, 0)
            Z = Z_rots(numb_photons, 0)
            psi = rot_uni_arb(3.5, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
            LC_ops = extract_LC_sequences([0, [1, 2, 3]], Z, X)
            X0, Z1, Z2, Z3 = LC_ops
            ops = [X0, [Z1, Z2, Z3]]
            psi = LC(psi, ops)
            LC_ops = extract_LC_sequences([1, [0, 2, 3]], Z, X)
            X1, Z0, Z2, Z3 = LC_ops
            ops = [X1, [Z0, Z2, Z3]]
            psi = LC(psi, ops)

        den = np.outer(psi, np.conjugate(psi).transpose())
        psi_sim += den / len(the_list)
    post_select = post_select_op(numb_photons)
    psi_sim = np.matmul(np.matmul(post_select, psi_sim), post_select)
    norm = np.trace(psi_sim)
    psi_sim = psi_sim / norm
    return psi_sim



if __name__ == "__main__":
    zero = np.array([1, 0, 0, 0])
    one = np.array([0, 1, 0, 0])
    early = np.array([0, 0, 1, 0])
    late = np.array([0, 1, 0, 0])
    mn = []
    for i in range(1):
        psi_sim = ghz(0.957, 0.0021, 23.2)  # three_star(0.947)
        # pure = (1 / np.sqrt(2)) * (np.kron(np.kron(zero, late), late) + (np.kron(np.kron(one, early), early)))
        pure = (1 / np.sqrt(2)) * (np.kron(zero, late) + np.kron(one, early))
        F = abs(np.matmul(np.matmul(pure, psi_sim), pure))
        print(F)
        print(np.trace(psi_sim))
        mn.append(F)

    print(np.mean(mn), np.std(mn))
    vacuum = np.array([0, 1, 0, 0])
    # pure = (1 / np.sqrt(2)) * ((np.kron(zero, vacuum)) + np.kron(one, vacuum))
    pure = np.kron(zero, vacuum)
    pure_density_matrix = np.outer(pure, np.conjugate(pure).transpose())
    inv = np.linalg.pinv(pure_density_matrix)
    # M = np.dot(psi_sim, inv)
    # print(M)
    M = psi_sim.dot(np.linalg.pinv(pure_density_matrix))
    new_transform = np.matmul(M, pure_density_matrix)
    # pure = (1 / np.sqrt(2)) * (np.kron(np.kron(zero, late), late) + (np.kron(np.kron(one, early), early)))
    pure = (1 / np.sqrt(2)) * (np.kron(zero, late) + np.kron(one, early))
    F_trans_new = abs(np.matmul(np.matmul(pure, new_transform), pure))
    print("new fid {}".format(F_trans_new))
    if np.array_equal(new_transform, psi_sim):
        print("hej")
    #print(new_transform[11])
    #print(psi_sim[11])
    #for i in range(23, 30):
    #    print("new {}".format(new_transform[i]))
    #    print("old{}".format(psi_sim[i]))
    #print("original {}".format(psi_sim))
    #print("new {}".format(new_transform))
    #M = np.kron(M, identity)
    #pure = (1 / np.sqrt(2)) * (np.kron(np.kron(zero, late), late) + (np.kron(np.kron(one, early), early)))
    #two_qubit_den = np.outer(pure, np.conjugate(pure).transpose())
    #two_qubit_den = np.matmul(M, two_qubit_den)
    #v, state = switch(pure, 2)
    #switched = (switch_density(two_qubit_den, v, v, 64))
    #two_qubit_den = np.matmul(M, switched)
    #two_qubit_den = (switch_density(two_qubit_den, v, v, 64))
    #F_trans_new = abs(np.matmul(np.matmul(pure, two_qubit_den), pure))
    #print("new fid {}".format(F_trans_new))






