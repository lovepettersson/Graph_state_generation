import numpy as np
import random
from tqdm import tqdm
from IndirectMeasurements.error_model_operators import LC, X_rots, Z_rots, extract_LC_sequences, \
    spin_flip_op, rot_uni_arb, switch, optical_collapse_operators,\
    excite_collapse_operators



def get_full_ops(numb_photons, ops):
    new_ops = []
    for op in ops:
        for i in range(numb_photons - 1):
            op = np.kron(op, identity)
        new_ops.append(op)
    return new_ops



def decay_op(psi, early_late, I, cyclicity, numb_photons):
    C = optical_collapse_operators(early_late, I, cyclicity)
    C = get_full_ops(numb_photons, C)
    P = []
    for c in C:
        psi_ket = np.matmul(c, psi)
        psi_bra = np.conjugate(psi_ket).transpose()
        p = np.matmul(psi_bra, psi_ket)
        P.append(p)
    choice = random.choices(range(len(P)), weights=P)[0]
    # choice = 0
    collapse = C[choice]
    prob = P[choice]
    psi = np.matmul(collapse, psi) / np.sqrt(prob)
    return psi


def excite_op(psi, numb_photons):
    C = excite_collapse_operators()
    C = get_full_ops(numb_photons, C)
    P = []
    for c in C:
        psi_ket = np.matmul(c, psi)
        psi_bra = np.conjugate(psi_ket).transpose()
        p = np.matmul(psi_bra, psi_ket)
        P.append(p)
    choice = random.choices(range(len(P)),  weights=P)[0]
    # choice = 0
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



def gen_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons):
    psi = excite_op(psi, numb_photons)
    psi = decay_op(psi, "early", I, C, numb_photons)
    psi = rot_uni_arb(7, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
    psi = excite_op(psi, numb_photons)
    psi = decay_op(psi, "late", I, C, numb_photons)
    psi = rot_uni_arb(7, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
    psi = np.matmul(H, psi)
    v, psi = switch(psi, numb_photons)
    return psi


def gen_con_photon(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons):
    psi = excite_op(psi, numb_photons)
    psi = decay_op(psi, "early", I, C, numb_photons)
    psi = rot_uni_arb(7, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
    psi = excite_op(psi, numb_photons)
    psi = decay_op(psi, "late", I, C, numb_photons)
    psi = rot_uni_arb(7, psi, 0.00000000000000001, numb_photons, C1, C2, deltaOH, 0)
    psi = np.matmul(H, psi)
    v, psi = switch(psi, numb_photons)
    return psi


def gen_line(I, kappa, T2, numb_photons):
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
    the_list = [i for i in range(2000)]
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
    the_list = [i for i in range(2000)]
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
