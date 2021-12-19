from gates import switch, QuditLateSPhoton, QuditEarlySPhoton, X_full_matrix, \
    X_half_matrix
from generate_graphs_and_spin_rot_op import spin_inverse_unitary, gen_new_photon, rot_uni_arb, spin_flip_op
import numpy as np
import matplotlib.pyplot as plt
from generate_graphs_and_spin_rot_op import LC_rotations_photons, spin_unitary, gen_GHZ, gates_state_sec, gen_GHZ_ideal


def gates_state_not_ideal(numb_photons, beta_par, beta_dia, delta0):
    qudit = np.array([1, 0, 0, 0], dtype=np.complex128)
    photon = np.array([1, 0, 0, 0], dtype=np.complex128)
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex128)

    delta2 = 0.9995  # 0.955, not including Delta00


    exc_matrix = np.array([[np.cos(np.pi/2*(delta0)), 0, 0, np.sin(np.pi/2*(delta0))],
                           [0, np.cos(np.pi/2*(delta2)), np.sin(np.pi/2*(delta2)), 0],
                           [0, -np.sin(np.pi/2*(delta2)), np.cos(np.pi/2*(delta2)), 0],
                           [-np.sin(np.pi/2*(delta0)), 0, 0, np.cos(np.pi/2*(delta0))],
                           ])


    H_matrix = np.array([[1, 0, 0, 0],
                         [0, 1 / (2 ** (1 / 2)), 1 / (2 ** (1 / 2)), 0],
                         [0, 1 / (2 ** (1 / 2)), -1 / (2 ** (1 / 2)), 0],
                         [0, 0, 0, 1],
                         ])

    H_gate_photon_1 = np.kron(identity, H_matrix)

    early_gate = QuditEarlySPhoton(beta_par, beta_dia).unitary()
    late_gate = QuditLateSPhoton(beta_par, beta_dia).unitary()

    ex = np.kron(exc_matrix, identity)
    collapse = spin_flip_op(1)
    C1 = collapse[0]
    C2 = collapse[1]
    state = np.kron(qudit, photon)
    C1 = np.kron(C1, identity)
    C2 = np.kron(C2, identity)


    for i in range(numb_photons - 1):
        state = np.kron(state, photon)
        ex = np.kron(ex, identity)
        early_gate = np.kron(early_gate, identity)
        late_gate = np.kron(late_gate, identity)
        C1 = np.kron(C1, identity)
        C2 = np.kron(C2, identity)
        H_gate_photon_1 = np.kron(H_gate_photon_1, identity)

    #state = np.matmul(pi_half_y_gate, state) # HERE IS THE INTIAL ROT FOR PERFECT STATE
    return ex, early_gate, late_gate, state, C1, C2, H_gate_photon_1


def LC_Z_spin(deltaOH):
    # THIS IS A SEPERATE FUNCTION BECAUSE WE ARE GOING TO SAMPLE THIS UNITARY DURING MC SIMULATION,
    # SPECIFICALLY DESIGN FOR DEUTSCH AND GROVER, I.E N=4.
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    Z_rot_spin = np.conjugate(spin_inverse_unitary(0, 0, np.pi / 2, 1 + deltaOH))
    Z_rot_spin = np.kron(np.kron(np.kron(np.kron(Z_rot_spin, identity), identity), identity), identity)
    return Z_rot_spin

def gates_state(numb_photons, beta_par, beta_dia, delta0):
    qudit = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], dtype=np.complex128)
    photon = np.array([1, 0, 0, 0], dtype=np.complex128)
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex128)

    X_inv = np.array([[1 / np.sqrt(2), 1j / np.sqrt(2), 0, 0],
                      [1j / np.sqrt(2), 1 / np.sqrt(2), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.complex128)

    exc_matrix = np.array([[np.sqrt(1 - delta0 ** 2), 0, 0, delta0],
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [-delta0, 0, 0, np.sqrt(1 - delta0 ** 2)],
                           ])
    H_matrix = np.array([[1, 0, 0, 0],
                         [0, 1 / (2 ** (1 / 2)), 1 / (2 ** (1 / 2)), 0],
                         [0, 1 / (2 ** (1 / 2)), -1 / (2 ** (1 / 2)), 0],
                         [0, 0, 0, 1],
                         ])

    H_gate_photon_1 = np.kron(np.kron(np.kron(np.kron(identity, H_matrix), identity), identity), identity)

    early_gate = QuditEarlySPhoton(beta_par, beta_dia).unitary()
    late_gate = QuditLateSPhoton(beta_par, beta_dia).unitary()
    ex = np.kron(exc_matrix, identity)
    X_half_perfect = np.kron(X_half_matrix, identity)
    X_full_perfect = np.kron(X_full_matrix, identity)
    state = np.kron(qudit, photon)

    for i in range(numb_photons - 1):
        state = np.kron(state, photon)
        ex = np.kron(ex, identity)
        early_gate = np.kron(early_gate, identity)
        late_gate = np.kron(late_gate, identity)
        X_half_perfect = np.kron(X_half_perfect, identity)
        X_full_perfect = np.kron(X_full_perfect, identity)


    return ex, early_gate, late_gate, X_full_perfect, X_half_perfect, state, H_gate_photon_1



def photon_rotations():
    square_root = 1 / (2 ** (1 / 2))
    rot_matrix = np.array([[1, 0, 0, 0],
                           [0, (1 + 1j) * square_root, 0, 0],
                           [0, 0, (1 - 1j) * square_root, 0],
                           [0, 0, 0, 1],
                           ])
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex128)
    Z_rot_gate_1 = np.kron(np.kron(np.kron(np.kron(identity, rot_matrix), identity), identity), identity)
    Z_rot_gate_2 = np.kron(np.kron(np.kron(np.kron(identity, identity), rot_matrix), identity), identity)
    Z_rot_gate_3 = np.kron(np.kron(np.kron(np.kron(identity, identity), identity), rot_matrix), identity)
    Z_rot_gate_4 = np.kron(np.kron(np.kron(np.kron(identity, identity), identity), identity), rot_matrix)
    rot_matrix = np.array([[1, 0, 0, 0],
                           [0, 1 * square_root, -1j * square_root, 0],
                           [0, -1j * square_root, 1 * square_root, 0],
                           [0, 0, 0, 1],
                           ])
    X_rot_gate_1 = np.kron(np.kron(np.kron(np.kron(identity, rot_matrix), identity), identity), identity)
    X_rot_gate_3 = np.kron(np.kron(np.kron(np.kron(identity, identity), identity), rot_matrix), identity)
    X_rot_gate_4 = np.kron(np.kron(np.kron(np.kron(identity, identity), identity), identity), rot_matrix)
    rot_matrix = np.array([[(1 + 1j) * square_root, 0, 0, 0],
                           [0, (1 - 1j) * square_root, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],
                           ])
    Z_rot_spin = np.kron(np.kron(np.kron(np.kron(rot_matrix, identity), identity), identity), identity)

    return Z_rot_gate_1, Z_rot_gate_2, Z_rot_gate_3, X_rot_gate_1, X_rot_gate_3, Z_rot_spin, X_rot_gate_4, Z_rot_gate_4


def LC_spin(state):
    pi_half_pulse = spin_unitary(0, np.pi / 7, 3.5, 0)
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex128)
    pi_half = np.kron(np.kron(np.kron(np.kron(pi_half_pulse, identity), identity), identity), identity)
    return np.matmul(pi_half, state)


def photo_gen_ideal(state, ex, early_gate, late_gate, X_full_perfect, H):
    state = np.matmul(ex, state)
    state = np.matmul(early_gate, state)
    state = np.matmul(X_full_perfect, state)
    state = np.matmul(ex, state)
    state = np.matmul(late_gate, state)
    state = np.matmul(X_full_perfect, state)
    state = np.matmul(H, state)

    return state




def gen_line(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, H, deltaOH):
    Z_rot_gate_1, Z_rot_gate_2, Z_rot_gate_3, X_rot_gate_1, X_rot_gate_3, Z_shit, X_rot_gate_4, Z_rot_gate_4 = photon_rotations()
    Z_rot_spin = LC_Z_spin(deltaOH)

    # INIT
    state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)

    # FIRST PHOTON
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)

    # SECOND PHOTON
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)

    # FIRST LC
    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)

    state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH, 0)
    state = np.matmul(Z_rot_gate_1, state)
    state = np.matmul(Z_rot_gate_2, state)

    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)

    # THIRD PHOTON
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)

    # SECOND LC
    v, state = switch(state, numb_photons)
    state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH, 0)
    state = np.matmul(Z_rot_gate_1, state)
    state = np.matmul(Z_rot_gate_2, state)
    state = np.matmul(Z_rot_gate_3, state)
    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)

    # FOURTH PHOTON
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)

    # THIRD LC
    state = np.matmul(X_rot_gate_3, state)
    state = np.matmul(Z_rot_gate_1, state)
    state = np.matmul(Z_rot_gate_2, state)
    state = np.matmul(Z_rot_spin, state)

    # FOURTH LC
    state = np.matmul(X_rot_gate_1, state)
    state = np.matmul(Z_rot_gate_2, state)
    state = np.matmul(Z_rot_gate_3, state)

    #FITH LC
    state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH, 0)
    state = np.matmul(Z_rot_gate_3, state)
    state = np.matmul(Z_rot_gate_4, state)

    #SIXTH LC
    state = np.matmul(X_rot_gate_4, state)
    state = np.matmul(Z_rot_spin, state)
    state = np.matmul(Z_rot_gate_3, state)

    return state


def generate_line_ideal(numb_photons):
    ex, early_gate, late_gate, X_full_perfect, X_half_perfect, state, H_gate_photon_1 = gates_state(
        numb_photons, 1, 0, 0)
    Z_rot_gate_1, Z_rot_gate_2, Z_rot_gate_3, X_rot_gate_1, X_rot_gate_3, Z_rot_spin, X_rot_gate_4, Z_rot_gate_4 = photon_rotations()

    # FIRST PHOTON
    state = photo_gen_ideal(state, ex, early_gate, late_gate, X_full_perfect, H_gate_photon_1)

    v, state = switch(state, numb_photons)
    # SECOND PHOTON
    state = photo_gen_ideal(state, ex, early_gate, late_gate, X_full_perfect, H_gate_photon_1)
    v, state = switch(state, numb_photons)

    # FIRST LC
    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)

    state = LC_spin(state)
    state = np.matmul(Z_rot_gate_1, state)
    state = np.matmul(Z_rot_gate_2, state)

    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)

    # THIRD PHOTON
    state = photo_gen_ideal(state, ex, early_gate, late_gate, X_full_perfect, H_gate_photon_1)
    v, state = switch(state, numb_photons)

    # SECOND LC
    v, state = switch(state, numb_photons)
    state = LC_spin(state)
    state = np.matmul(Z_rot_gate_1, state)
    state = np.matmul(Z_rot_gate_2, state)
    state = np.matmul(Z_rot_gate_3, state)
    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)

    # FOURTH PHOTON
    state = photo_gen_ideal(state, ex, early_gate, late_gate, X_full_perfect, H_gate_photon_1)
    v, state = switch(state, numb_photons)

    # THIRD LC
    state = np.matmul(X_rot_gate_3, state)
    state = np.matmul(Z_rot_gate_1, state)
    state = np.matmul(Z_rot_gate_2, state)
    state = np.matmul(Z_rot_spin, state)

    # FOURTH LC
    state = np.matmul(X_rot_gate_1, state)
    state = np.matmul(Z_rot_gate_2, state)
    state = np.matmul(Z_rot_gate_3, state)

    '''
    #FITH LC
    state = LC_spin(state)
    state = np.matmul(Z_rot_gate_3, state)
    state = np.matmul(Z_rot_gate_4, state)

    #SIXTH LC
    state = np.matmul(X_rot_gate_4, state)
    state = np.matmul(Z_rot_gate_3, state)
    state = np.matmul(Z_rot_spin, state)

    #state = state * 1j# overall phase, not needed
    '''
    return state, X_full_perfect, Z_rot_spin



def logical_pauli():

    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex128)

    X_spin = np.array([[0, 1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex128)

    X_photon = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]], dtype=np.complex128)

    Z_spin = np.array([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=np.complex128)

    Z_photon = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]], dtype=np.complex128)

    Y_spin = np.array([[0, -1j, 0, 0],
                       [1j, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=np.complex128)

    Y_photon = np.array([[1, 0, 0, 0],
                       [0, 0, 1j, 0],
                       [0, -1j, 0, 0],
                       [0, 0, 0, 1]], dtype=np.complex128)

    Z_photon = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]], dtype=np.complex128)

    photon_Z_up = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=np.complex128) # Change to Z
    photon_Z_up = np.outer(photon_Z_up, np.conjugate(photon_Z_up).transpose())

    logical_Z = np.kron(np.kron(np.kron(np.kron(X_spin, identity), identity), Z_photon), X_photon)
    logical_Y = np.kron(np.kron(np.kron(np.kron(X_spin, identity), identity), Z_photon), Y_photon)
    logical_X = np.kron(np.kron(np.kron(np.kron(X_spin, identity), identity), identity), Z_photon)
    logical_I = np.kron(np.kron(np.kron(np.kron(X_spin, identity), identity), identity), identity)

    return logical_Z, logical_Y, logical_X, logical_I


def projector_GHZ():
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex128)

    Z_photon = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1]], dtype=np.complex128)

    Y_photon = np.array([[1, 0, 0, 0],
                         [0, 0, -1j, 0],
                         [0, 1j, 0, 0],
                         [0, 0, 0, 1]], dtype=np.complex128)

    X_photon = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=np.complex128)

    photon_X_up = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=np.complex128)
    photon_Z_up = np.array([0, 1, 0, 0], dtype=np.complex128)
    photon_Z_down = np.array([0, 0, 1, 0], dtype=np.complex128)
    photon_Y_up = np.array([0, 1 / np.sqrt(2), -1j / np.sqrt(2), 0], dtype=np.complex128)
    spin_Z_up = np.array([1, 0, 0, 0], dtype=np.complex128)
    spin_X_up = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], dtype=np.complex128) # ugly fix
    spin_Y_up = np.array([1 / np.sqrt(2), 1j / np.sqrt(2), 0, 0], dtype=np.complex128)
    photon_X_up = np.outer(photon_X_up, photon_X_up.transpose())
    spin_X_up = np.outer(spin_X_up, spin_X_up.transpose())
    spin_Z_up = np.outer(spin_Z_up, spin_Z_up.transpose())
    spin_Y_up = np.outer(spin_Y_up, spin_Y_up.transpose())
    photon_Z_up = np.outer(photon_Z_up, np.conjugate(photon_Z_up).transpose())
    photon_Z_down = np.outer(photon_Z_down, np.conjugate(photon_Z_down).transpose())
    photon_Y_up = np.outer(photon_Y_up, np.conjugate(photon_Y_up).transpose())

    X_up_photon = np.kron(identity, photon_X_up)
    Z_up_photon = np.kron(identity, photon_Z_up)
    Z_down_photon = np.kron(identity, photon_Z_down)
    Y_up_photon = np.kron(identity, photon_Y_up)
    X_up_spin = np.kron(spin_X_up, identity)
    Z_up_spin = np.kron(spin_Z_up, identity)
    Y_up_spin = np.kron(spin_Y_up, identity)
    Z_photon = np.kron(spin_X_up, Z_photon)
    Y_photon = np.kron(spin_X_up, Y_photon)
    X_photon = np.kron(spin_X_up, X_photon)
    I_photon = np.kron(spin_X_up, identity)

    return X_up_spin, X_up_photon, Y_up_spin, Z_up_spin, Z_up_photon, Z_down_photon, Y_up_photon, Z_photon, Y_photon, X_photon, I_photon


def projectors():
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex128)
    photon_Z_up = np.array([0, 1, 0, 0], dtype=np.complex128)
    photon_Z_down = np.array([0, 0, 1, 0], dtype=np.complex128)
    photon_X_up = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=np.complex128)
    photon_X_down = np.array([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0], dtype=np.complex128)
    spin_Z_up = np.array([1, 0, 0, 0], dtype=np.complex128)
    spin_Z_down = np.array([0, 1, 0, 0], dtype=np.complex128)
    spin_X_up = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], dtype=np.complex128)
    spin_X_down = np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0, 0], dtype=np.complex128)
    spin_Y_up = np.array([1 / np.sqrt(2), 1j / np.sqrt(2), 0, 0], dtype=np.complex128)
    spin_Y_down = np.array([1 / np.sqrt(2), -1j / np.sqrt(2), 0, 0], dtype=np.complex128)

    photon_Z_up = np.outer(photon_Z_up, photon_Z_up.transpose())
    photon_Z_down = np.outer(photon_Z_down, photon_Z_down.transpose())
    photon_X_up = np.outer(photon_X_up, photon_X_up.transpose())
    photon_X_down = np.outer(photon_X_down, photon_X_down.transpose())
    spin_Z_up = np.outer(spin_Z_up, spin_Z_up.transpose())
    spin_Z_down = np.outer(spin_Z_down, spin_Z_down.transpose())
    spin_X_down = np.outer(spin_X_down, spin_X_down.transpose())
    spin_X_up = np.outer(spin_X_up, spin_X_up.transpose())
    spin_Y_down = np.outer(spin_Y_down, spin_Y_down.transpose())
    spin_Y_up = np.outer(spin_Y_up, spin_Y_up.transpose())

    Z_up_4 = np.kron(np.kron(np.kron(np.kron(identity, identity), identity), identity), photon_Z_up) # ugly trick
    Z_down_4 = np.kron(np.kron(np.kron(np.kron(identity, identity), identity), identity), photon_Z_down)
    X_up_2 = np.kron(np.kron(np.kron(np.kron(identity, identity), photon_X_up), identity), identity)
    X_down_2 = np.kron(np.kron(np.kron(np.kron(identity, identity), photon_X_down), identity), identity)
    X_up_3 = np.kron(np.kron(np.kron(np.kron(identity, identity), identity), photon_X_up), identity)
    X_down_3 = np.kron(np.kron(np.kron(np.kron(identity, identity), identity), photon_X_down), identity)
    Z_up_1 = np.kron(np.kron(np.kron(np.kron(identity, photon_Z_up), identity), identity), identity)
    Z_down_1 = np.kron(np.kron(np.kron(np.kron(identity, photon_Z_down), identity), identity), identity)
    Z_up_3 = np.kron(np.kron(np.kron(np.kron(identity, identity), identity), photon_Z_up), identity)
    Z_down_3 = np.kron(np.kron(np.kron(np.kron(identity, identity), identity), photon_Z_down), identity)
    Z_up_spin = np.kron(np.kron(np.kron(np.kron(spin_Z_up, identity), identity), identity), identity)
    Z_down_spin = np.kron(np.kron(np.kron(np.kron(spin_Z_down, identity), identity), identity), identity)
    X_up_spin = np.kron(np.kron(np.kron(np.kron(spin_X_up, identity), identity), identity), identity)
    X_down_spin = np.kron(np.kron(np.kron(np.kron(spin_X_down, identity), identity), identity), identity)
    Y_up_spin = np.kron(np.kron(np.kron(np.kron(spin_Y_up, identity), identity), identity), identity)
    Y_down_spin = np.kron(np.kron(np.kron(np.kron(spin_Y_down, identity), identity), identity), identity)

    return Z_up_spin, Z_down_spin, Z_up_1, Z_down_1, X_up_2, X_down_2, X_up_3, X_down_3, Z_up_4, Z_down_4, X_up_spin, X_down_spin, Y_up_spin, Y_down_spin, Z_up_3, Z_down_3


def prob_den(operator, den_matrix):
    projection = np.matmul(operator, np.matmul(den_matrix, np.conjugate(operator).transpose()))
    prob = np.trace(projection)
    #print(prob)
    projection = projection / prob
    return prob, projection


def probabilities_den(Z_up_spin, Z_down_spin, Z_up_1, Z_down_1, X_up_2, X_down_2, X_up_3, X_down_3, state):

    # STABILIZER 3
    prob_chain_plus_S = {}
    prob_chain_minus_S = {}
    den_matrix = state
    prob_Z_up_S, plus_S = prob_den(Z_up_spin, den_matrix)

    prob_chain_plus_S["P_+S"] = (prob_Z_up_S)

    # MEAS 3 PLUS S
    prob_X_up, plus_3 = prob_den(X_up_3, plus_S)
    prob_X_down, minus_3 = prob_den(X_down_3, plus_S)
    #prob_X_up, prob_X_down, plus_3, minus_3 = measure_3(plus_S, X_up_3, X_down_3)
    prob_chain_plus_S["P_+3"] = (prob_X_up)
    prob_chain_plus_S["P_-3"] = (prob_X_down)

    # MEAS 1 PLUS 3
    prob_Z_up, plus_3_plus_1 = prob_den(Z_up_1, plus_3)
    prob_Z_down, plus_3_minus_1 = prob_den(Z_down_1, plus_3)
    #prob_Z_up, prob_Z_down, plus_3_plus_1, plus_3_minus_1 = measure_1(plus_3, Z_up_1, Z_down_1)
    prob_chain_plus_S["P_+3_+1"] = (prob_Z_up)
    prob_chain_plus_S["P_+3_-1"] = (prob_Z_down)

    # MEAS 1 MINUS 3
    prob_Z_up, minus_3_plus_1 = prob_den(Z_up_1, minus_3)
    prob_Z_down, minus_3_minus_1 = prob_den(Z_down_1, minus_3)
    #prob_Z_up, prob_Z_down, minus_3_plus_1, minus_3_minus_1 = measure_1(minus_3, Z_up_1, Z_down_1)
    prob_chain_plus_S["P_-3_+1"] = (prob_Z_up)
    prob_chain_plus_S["P_-3_-1"] = (prob_Z_down)


    # MEAS 2 PLUS 3 PLUS 1
    prob_X_up, plus_3_plus_1_plus_2 = prob_den(X_up_2, plus_3_plus_1)
    prob_X_down, plus_3_plus_1_minus_2 = prob_den(X_down_2, plus_3_plus_1)
    #prob_X_up, prob_X_down, plus_3_plus_1_plus_2, plus_3_plus_1_minus_2 = measure_2(plus_3_plus_1, X_up_2, X_down_2)
    prob_chain_plus_S["P_+3_+1_+2"] = (prob_X_up)
    prob_chain_plus_S["P_+3_+1_-2"] = (prob_X_down)

    # MEAS 2 PLUS 3 MINUS 1
    prob_X_up, plus_3_plus_1_plus_2 = prob_den(X_up_2, plus_3_minus_1)
    prob_X_down, plus_3_plus_1_minus_2 = prob_den(X_down_2, plus_3_minus_1)
    #prob_X_up, prob_X_down, plus_3_minus_1_plus_2, plus_3_minus_1_minus_2 = measure_2(plus_3_minus_1, X_up_2, X_down_2)
    prob_chain_plus_S["P_+3_-1_+2"] = (prob_X_up)
    prob_chain_plus_S["P_+3_-1_-2"] = (prob_X_down)

    # MEAS 2 MINUS 3 PLUS 1
    prob_X_up, plus_3_plus_1_plus_2 = prob_den(X_up_2, minus_3_plus_1)
    prob_X_down, plus_3_plus_1_minus_2 = prob_den(X_down_2, minus_3_plus_1)
    #prob_X_up, prob_X_down, minus_3_plus_1_plus_2, minus_3_plus_1_minus_2 = measure_2(minus_3_plus_1, X_up_2, X_down_2)
    prob_chain_plus_S["P_-3_+1_+2"] = (prob_X_up)
    prob_chain_plus_S["P_-3_+1_-2"] = (prob_X_down)

    # MEAS 2 MINUS 3 MINUS 1
    prob_X_up, plus_3_plus_1_plus_2 = prob_den(X_up_2, minus_3_minus_1)
    prob_X_down, plus_3_plus_1_minus_2 = prob_den(X_down_2, minus_3_minus_1)
    #prob_X_up, prob_X_down, minus_3_minus_1_plus_2, minus_3_minus_1_minus_2 = measure_2(minus_3_minus_1, X_up_2, X_down_2)
    prob_chain_plus_S["P_-3_-1_+2"] = (prob_X_up)
    prob_chain_plus_S["P_-3_-1_-2"] = (prob_X_down)

    # MINUS S
    prob_Z_down_S, minus_S = prob_den(Z_down_spin, den_matrix)
    #prob_Z_down_S = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(Z_down_spin, state)))
    #minus_S = np.matmul(Z_down_spin, state) / np.sqrt(prob_Z_up_S)
    prob_chain_minus_S["P_-S"] = prob_Z_down_S

    # MEAS 3 MINUS S
    prob_X_up, plus_3 = prob_den(X_up_3, minus_S)
    prob_X_down, minus_3 = prob_den(X_down_3, minus_S)
    #prob_X_up, prob_X_down, plus_3, minus_3 = measure_3(minus_S, X_up_3, X_down_3)
    prob_chain_minus_S["P_+3"] = (prob_X_up)
    prob_chain_minus_S["P_-3"] = (prob_X_down)

    # MEAS 1 PLUS 3
    prob_Z_up, plus_3_plus_1 = prob_den(Z_up_1, plus_3)
    prob_Z_down, plus_3_minus_1 = prob_den(Z_down_1, plus_3)
    #prob_Z_up, prob_Z_down, plus_3_plus_1, plus_3_minus_1 = measure_1(plus_3, Z_up_1, Z_down_1)
    prob_chain_minus_S["P_+3_+1"] = (prob_Z_up)
    prob_chain_minus_S["P_+3_-1"] = (prob_Z_down)

    # MEAS 1 MINUS 3
    prob_Z_up, minus_3_plus_1 = prob_den(Z_up_1, minus_3)
    prob_Z_down, minus_3_minus_1 = prob_den(Z_down_1, minus_3)
    #prob_Z_up, prob_Z_down, minus_3_plus_1, minus_3_minus_1 = measure_1(minus_3, Z_up_1, Z_down_1)
    prob_chain_minus_S["P_-3_+1"] = (prob_Z_up)
    prob_chain_minus_S["P_-3_-1"] = (prob_Z_down)

    # MEAS 2 PLUS 3 PLUS 1
    prob_X_up, plus_3_plus_1_plus_2 = prob_den(X_up_2, plus_3_plus_1)
    prob_X_down, plus_3_plus_1_minus_2 = prob_den(X_down_2, plus_3_plus_1)
    #prob_X_up, prob_X_down, plus_3_plus_1_plus_2, plus_3_plus_1_minus_2 = measure_2(plus_3_plus_1, X_up_2, X_down_2)
    prob_chain_minus_S["P_+3_+1_+2"] = (prob_X_up)
    prob_chain_minus_S["P_+3_+1_-2"] = (prob_X_down)

    # MEAS 2 PLUS 3 MINUS 1
    prob_X_up, plus_3_plus_1_plus_2 = prob_den(X_up_2, plus_3_minus_1)
    prob_X_down, plus_3_plus_1_minus_2 = prob_den(X_down_2, plus_3_minus_1)
    #prob_X_up, prob_X_down, plus_3_minus_1_plus_2, plus_3_minus_1_minus_2 = measure_2(plus_3_minus_1, X_up_2, X_down_2)
    prob_chain_minus_S["P_+3_-1_+2"] = (prob_X_up)
    prob_chain_minus_S["P_+3_-1_-2"] = (prob_X_down)

    # MEAS 2 MINUS 3 PLUS 1
    prob_X_up, plus_3_plus_1_plus_2 = prob_den(X_up_2, minus_3_plus_1)
    prob_X_down, plus_3_plus_1_minus_2 = prob_den(X_down_2, minus_3_plus_1)
    #prob_X_up, prob_X_down, minus_3_plus_1_plus_2, minus_3_plus_1_minus_2 = measure_2(minus_3_plus_1, X_up_2, X_down_2)
    prob_chain_minus_S["P_-3_+1_+2"] = (prob_X_up)
    prob_chain_minus_S["P_-3_+1_-2"] = (prob_X_down)

    # MEAS 2 MINUS 3 MINUS 1
    prob_X_up, plus_3_plus_1_plus_2 = prob_den(X_up_2, minus_3_minus_1)
    prob_X_down, plus_3_plus_1_minus_2 = prob_den(X_down_2, minus_3_minus_1)
    #prob_X_up, prob_X_down, minus_3_minus_1_plus_2, minus_3_minus_1_minus_2 = measure_2(minus_3_minus_1, X_up_2,
     #                                                                                   X_down_2)
    prob_chain_minus_S["P_-3_-1_+2"] = (prob_X_up)
    prob_chain_minus_S["P_-3_-1_-2"] = (prob_X_down)



    return prob_chain_plus_S, prob_chain_minus_S


def probabilities(Z_up_spin, Z_down_spin, Z_up_1, Z_down_1, X_up_2, X_down_2, X_up_3, X_down_3, state):

    # STABILIZER 3
    prob_chain_plus_S = {}
    prob_chain_minus_S = {}

    prob_Z_up_S = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(Z_up_spin, state)))
    plus_S = np.matmul(Z_up_spin, state) / np.sqrt(prob_Z_up_S)
    prob_chain_plus_S["P_+S"] = (prob_Z_up_S)

    # MEAS 3 PLUS S
    prob_X_up, prob_X_down, plus_3, minus_3 = measure_3(plus_S, X_up_3, X_down_3)
    prob_chain_plus_S["P_+3"] = (prob_X_up)
    prob_chain_plus_S["P_-3"] = (prob_X_down)

    # MEAS 1 PLUS 3
    prob_Z_up, prob_Z_down, plus_3_plus_1, plus_3_minus_1 = measure_1(plus_3, Z_up_1, Z_down_1)
    prob_chain_plus_S["P_+3_+1"] = (prob_Z_up)
    prob_chain_plus_S["P_+3_-1"] = (prob_Z_down)

    # MEAS 1 MINUS 3
    prob_Z_up, prob_Z_down, minus_3_plus_1, minus_3_minus_1 = measure_1(minus_3, Z_up_1, Z_down_1)
    prob_chain_plus_S["P_-3_+1"] = (prob_Z_up)
    prob_chain_plus_S["P_-3_-1"] = (prob_Z_down)


    # MEAS 2 PLUS 3 PLUS 1
    prob_X_up, prob_X_down, plus_3_plus_1_plus_2, plus_3_plus_1_minus_2 = measure_2(plus_3_plus_1, X_up_2, X_down_2)
    prob_chain_plus_S["P_+3_+1_+2"] = (prob_X_up)
    prob_chain_plus_S["P_+3_+1_-2"] = (prob_X_down)

    # MEAS 2 PLUS 3 MINUS 1
    prob_X_up, prob_X_down, plus_3_minus_1_plus_2, plus_3_minus_1_minus_2 = measure_2(plus_3_minus_1, X_up_2, X_down_2)
    prob_chain_plus_S["P_+3_-1_+2"] = (prob_X_up)
    prob_chain_plus_S["P_+3_-1_-2"] = (prob_X_down)

    # MEAS 2 MINUS 3 PLUS 1
    prob_X_up, prob_X_down, minus_3_plus_1_plus_2, minus_3_plus_1_minus_2 = measure_2(minus_3_plus_1, X_up_2, X_down_2)
    prob_chain_plus_S["P_-3_+1_+2"] = (prob_X_up)
    prob_chain_plus_S["P_-3_+1_-2"] = (prob_X_down)

    # MEAS 2 MINUS 3 MINUS 1
    prob_X_up, prob_X_down, minus_3_minus_1_plus_2, minus_3_minus_1_minus_2 = measure_2(minus_3_minus_1, X_up_2, X_down_2)
    prob_chain_plus_S["P_-3_-1_+2"] = (prob_X_up)
    prob_chain_plus_S["P_-3_-1_-2"] = (prob_X_down)

    # MINUS S
    prob_Z_down_S = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(Z_down_spin, state)))
    minus_S = np.matmul(Z_down_spin, state) / np.sqrt(prob_Z_up_S)
    prob_chain_minus_S["P_-S"] = prob_Z_down_S

    # MEAS 3 MINUS S
    prob_X_up, prob_X_down, plus_3, minus_3 = measure_3(minus_S, X_up_3, X_down_3)
    prob_chain_minus_S["P_+3"] = (prob_X_up)
    prob_chain_minus_S["P_-3"] = (prob_X_down)

    # MEAS 1 PLUS 3
    prob_Z_up, prob_Z_down, plus_3_plus_1, plus_3_minus_1 = measure_1(plus_3, Z_up_1, Z_down_1)
    prob_chain_minus_S["P_+3_+1"] = (prob_Z_up)
    prob_chain_minus_S["P_+3_-1"] = (prob_Z_down)

    # MEAS 1 MINUS 3
    prob_Z_up, prob_Z_down, minus_3_plus_1, minus_3_minus_1 = measure_1(minus_3, Z_up_1, Z_down_1)
    prob_chain_minus_S["P_-3_+1"] = (prob_Z_up)
    prob_chain_minus_S["P_-3_-1"] = (prob_Z_down)

    # MEAS 2 PLUS 3 PLUS 1
    prob_X_up, prob_X_down, plus_3_plus_1_plus_2, plus_3_plus_1_minus_2 = measure_2(plus_3_plus_1, X_up_2, X_down_2)
    prob_chain_minus_S["P_+3_+1_+2"] = (prob_X_up)
    prob_chain_minus_S["P_+3_+1_-2"] = (prob_X_down)

    # MEAS 2 PLUS 3 MINUS 1
    prob_X_up, prob_X_down, plus_3_minus_1_plus_2, plus_3_minus_1_minus_2 = measure_2(plus_3_minus_1, X_up_2, X_down_2)
    prob_chain_minus_S["P_+3_-1_+2"] = (prob_X_up)
    prob_chain_minus_S["P_+3_-1_-2"] = (prob_X_down)

    # MEAS 2 MINUS 3 PLUS 1
    prob_X_up, prob_X_down, minus_3_plus_1_plus_2, minus_3_plus_1_minus_2 = measure_2(minus_3_plus_1, X_up_2, X_down_2)
    prob_chain_minus_S["P_-3_+1_+2"] = (prob_X_up)
    prob_chain_minus_S["P_-3_+1_-2"] = (prob_X_down)

    # MEAS 2 MINUS 3 MINUS 1
    prob_X_up, prob_X_down, minus_3_minus_1_plus_2, minus_3_minus_1_minus_2 = measure_2(minus_3_minus_1, X_up_2,
                                                                                        X_down_2)
    prob_chain_minus_S["P_-3_-1_+2"] = (prob_X_up)
    prob_chain_minus_S["P_-3_-1_-2"] = (prob_X_down)



    return prob_chain_plus_S, prob_chain_minus_S


def measure_3(state, X_up_3, X_down_3):
    prob_X_plus_3 = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(X_up_3, state)))
    plus_3 = np.matmul(X_up_3, state) / np.sqrt(prob_X_plus_3)

    prob_X_down_3 = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(X_down_3, state)))
    minus_3 = np.matmul(X_down_3, state) / np.sqrt(prob_X_down_3)

    return prob_X_plus_3, prob_X_down_3, plus_3, minus_3


def measure_1(state, Z_up_1, Z_down_1):
    prob_Z_up_1 = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(Z_up_1, state)))
    plus_1 = np.matmul(Z_up_1, state) / np.sqrt(prob_Z_up_1)

    prob_Z_down_1 = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(Z_down_1, state)))
    minus_1 = np.matmul(Z_down_1, state) / np.sqrt(prob_Z_down_1)

    return prob_Z_up_1, prob_Z_down_1, plus_1, minus_1


def measure_2(state, X_up_2, X_down_2):
    prob_X_up_2 = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(X_up_2, state)))
    plus_2 = np.matmul(X_up_2, state) / np.sqrt(prob_X_up_2)

    prob_X_down_2 = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(X_down_2, state)))
    minus_2 = np.matmul(X_down_2, state) / np.sqrt(prob_X_up_2)

    return prob_X_up_2, prob_X_down_2, plus_2, minus_2

def rules_switching(prob_S_plus, prob_S_minus):
    P_S_p = prob_S_plus["P_+S"]
    P_3_p = prob_S_plus["P_+3"]
    P_1_p = prob_S_plus["P_+3_+1"]
    P_2_p = prob_S_plus["P_+3_+1_+2"]
    P_1_m_2_p = prob_S_plus["P_+3_-1_+2"]
    P_3_m_1_p = prob_S_plus["P_-3_+1"]
    P_3_m_1_p_2_p = prob_S_plus["P_-3_+1_+2"]
    P_3_m_1_m_2_p = prob_S_plus["P_-3_-1_+2"]
    X = 0
    S3 = 1
    S2 = 1
    if P_S_p > np.random.uniform():
        #print("S+")
        X = 1
        if P_3_p > np.random.uniform():
            S3 = S3 * 1
            if P_1_p > np.random.uniform():
                S3 = S3 * 1
                S2 = S2 * 1
                if P_2_p > np.random.uniform():
                    S2 = S2 * 1
                else:
                    S2 = S2 * (-1)
            else:
                S3 = S3 * (-1)
                S2 = S2 * (-1)
                if P_1_m_2_p > np.random.uniform():
                    S2 = S2 * 1
                else:
                    S2 = S2 * (-1)
        else:
            S3 = S3 * (-1)
            if P_3_m_1_p > np.random.uniform():
                S3 = S3 * 1
                S2 = S2 * 1
                if P_3_m_1_p_2_p > np.random.uniform():
                    S2 = S2 * 1
                else:
                    S2 = S2 * (-1)
            else:
                S3 = S3 * (-1)
                S2 = S2 * (-1)
                if P_3_m_1_m_2_p > np.random.uniform():
                    S2 = S2 * 1
                else:
                    S2 = S2 * (-1)
    else:
        #print("S-")
        P_3_p = prob_S_minus["P_+3"]
        P_1_p = prob_S_minus["P_+3_+1"]
        P_2_p = prob_S_minus["P_+3_+1_+2"]
        P_1_m_2_p = prob_S_minus["P_+3_-1_+2"]
        P_3_m_1_p = prob_S_minus["P_-3_+1"]
        P_3_m_1_p_2_p = prob_S_minus["P_-3_+1_+2"]
        P_3_m_1_m_2_p = prob_S_minus["P_-3_-1_+2"]
        X = -1
        S3 = S3 * (-1)
        if P_3_p > np.random.uniform():
            S3 = S3 * 1
            if P_1_p > np.random.uniform():
                S3 = S3 * 1
                S2 = S2 * 1
                if P_2_p > np.random.uniform():
                    S2 = S2 * 1
                else:
                    S2 = S2 * (-1)
            else:
                S3 = S3 * (-1)
                S2 = S2 * (-1)
                if P_1_m_2_p > np.random.uniform():
                    S2 = S2 * 1
                else:
                    S2 = S2 * (-1)
        else:
            S3 = S3 * (-1)
            if P_3_m_1_p > np.random.uniform():
                S3 = S3 * 1
                S2 = S2 * 1
                if P_3_m_1_p_2_p > np.random.uniform():
                    S2 = S2 * 1
                else:
                    S2 = S2 * (-1)
            else:
                S3 = S3 * (-1)
                S2 = S2 * (-1)
                if P_3_m_1_m_2_p > np.random.uniform():
                    S2 = S2 * 1
                else:
                    S2 = S2 * (-1)
    '''
    if S3 == -1 and S2 == 1:
        if X == -1:
            X = 1
        else:
            X = -1
    '''
    #print("S3 {}, S2 {} and X {}".format(S3, S2, X))
    return X

def retrive_Z_logical_den(X_spin_up, X_spin_down, Z_up_3, Z_down_3, state):
    P_p_s, P_S = prob_den(X_up_spin, state)
    P_s_s, S_S = prob_den(X_down_spin, state)
    #P_p_s = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(X_spin_up, state)))
    Z = 1
    if P_p_s > np.random.uniform():
        Z = Z * 1
        state = P_S
        P_p_3, P_3 = prob_den(Z_up_3, state)
        #P_p_3 = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(Z_up_3, state)))
        if P_p_3 > np.random.uniform():
            Z = Z * 1
        else:
            Z = Z * (-1)
    else:
        Z = Z * (-1)
        state = S_S
        #state = np.matmul(X_spin_down, state) / np.sqrt(1- P_p_s)
        P_p_3, P_3 = prob_den(Z_up_3, state)
        #P_p_3 = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(Z_up_3, state)))
        if P_p_3 > np.random.uniform():
            Z = Z * 1
        else:
            Z = Z * (-1)
    return Z


def retrive_Z_logical(X_spin_up, X_spin_down, Z_up_3, Z_down_3, state):
    P_p_s = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(X_spin_up, state)))
    Z = 1
    if P_p_s > np.random.uniform():
        Z = Z * 1
        state = np.matmul(X_spin_up, state) / np.sqrt(P_p_s)
        P_p_3 = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(Z_up_3, state)))
        if P_p_3 > np.random.uniform():
            Z = Z * 1
        else:
            Z = Z * (-1)
    else:
        Z = Z * (-1)
        state = np.matmul(X_spin_down, state) / np.sqrt(1- P_p_s)
        P_p_3 = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(Z_up_3, state)))
        if P_p_3 > np.random.uniform():
            Z = Z * 1
        else:
            Z = Z * (-1)
    return Z


def retrive_Y_logical_den(Y_spin_up, Y_spin_down, Z_up_3, Z_down_3, state):
    P_p_s, P_S = prob_den(Y_spin_up, state)
    P_s_s, S_S = prob_den(Y_spin_down, state)
    #P_p_s = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(Y_spin_up, state)))
    Y = 1
    if P_p_s > np.random.uniform():
        Y = Y * 1
        state = P_S
        #state = np.matmul(Y_spin_up, state) / np.sqrt(P_p_s)
        P_p_3, P_3 = prob_den(Z_up_3, state)
        #P_p_3 = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(Z_up_3, state)))
        if P_p_3 > np.random.uniform():
            Y = Y * 1
        else:
            Y = Y * (-1)
    else:
        Y = Y * (-1)
        state = S_S
        #state = np.matmul(Y_spin_down, state) / np.sqrt(1 - P_p_s)
        P_p_3, P_3 = prob_den(Z_up_3, state)
        #P_p_3 = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(Z_up_3, state)))
        if P_p_3 > np.random.uniform():
            Y = Y * 1
        else:
            Y = Y * (-1)
    return Y


def retrive_Y_logical(Y_spin_up, Y_spin_down, Z_up_3, Z_down_3, state):
    P_p_s = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(Y_spin_up, state)))
    Y = 1
    if P_p_s > np.random.uniform():
        Y = Y * 1
        state = np.matmul(Y_spin_up, state) / np.sqrt(P_p_s)
        P_p_3 = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(Z_up_3, state)))
        if P_p_3 > np.random.uniform():
            Y = Y * 1
        else:
            Y = Y * (-1)
    else:
        Y = Y * (-1)
        state = np.matmul(Y_spin_down, state) / np.sqrt(1 - P_p_s)
        P_p_3 = abs(np.matmul(np.conjugate(state).transpose(), np.matmul(Z_up_3, state)))
        if P_p_3 > np.random.uniform():
            Y = Y * 1
        else:
            Y = Y * (-1)
    return Y


def GHZ_fid(kappas, Logical_M, T2):
    numb_photons = 1
    beta_par = 1 # 0.992
    beta_ort = 0 # 0.072
    cross_ex = 0 # 0.1
    X_up_s, X_up_photon, Y_up_s, Z_up_s, Z_up_photon, Z_down_photon, Y_up_photon, Z_photon, Y_photon, X_photon, I_photon = projector_GHZ()

    ex, ex_perfect, early_gate, late_gate, state, C1, C2, pi_half_gate, pi_gate, \
    early_gate_perfect, late_gate_perfect, H_gate_photon_1, \
    pi_half_y_gate = gates_state_sec(numb_photons, beta_par, beta_ort, cross_ex)

    M_list = []
    deltaOH = np.random.normal(0, np.sqrt(2) / T2)
    M = 0
    den_matrix = 0
    kappa = kappas
    for i in range(2000):
        not_ideal = gen_GHZ(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2,
                            H_gate_photon_1, deltaOH)
        den_matrix += np.outer(not_ideal, np.conjugate(not_ideal).transpose()) / 2000
    if Logical_M == "X":
        P, den_matrix = prob_den(X_up_s, den_matrix)
        P, den_matrix = prob_den(X_up_photon, den_matrix)
    elif Logical_M == "Z":
        P, den_matrix = prob_den(Z_up_s, den_matrix)
        P, den_matrix = prob_den(Z_up_photon, den_matrix)
        # P_up, den_matrix0 = prob_den(Z_up_photon, den_matrix)
        # P_down, den_matrix1 = prob_den(Z_down_photon, den_matrix)
        # P = P_up / (P_up + P_down)
    else:
        P, den_matrix = prob_den(Y_up_s, den_matrix)
        P, den_matrix = prob_den(Y_up_photon, den_matrix)
    for j in range(20000):
        if P > np.random.uniform():
            #  M += 1 / 2000
            continue
        else:
            #  M -= 1 / 2000
            M += 1 / 20000
    return M
    '''
    for kappa in kappas:
        M = 0
        den_matrix = 0
        for i in range(2000):
            not_ideal = gen_GHZ(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2,
                                H_gate_photon_1, deltaOH)
            den_matrix += np.outer(not_ideal, np.conjugate(not_ideal).transpose()) / 2000
        if Logical_M == "X":
            P, den_matrix = prob_den(X_up_s, den_matrix)
            P, den_matrix = prob_den(X_up_photon, den_matrix)
        elif Logical_M == "Z":
            P, den_matrix = prob_den(Z_up_s, den_matrix)
            P, den_matrix = prob_den(Z_up_photon, den_matrix)
        else:
            P, den_matrix = prob_den(Y_up_s, den_matrix)
            P, den_matrix = prob_den(Y_up_photon, den_matrix)
        for j in range(2000):
            if P > np.random.uniform():
                #  M += 1 / 2000
                continue
            else:
                #  M -= 1 / 2000
                M += 1/ 2000
        M_list.append(M)
        return M_list
        '''



if __name__ == "__main__":
    kappas = [0.00021, 5 * 0.00021, 0.0021, 5 * 0.0021, 0.021, 0.021 * 5, 0.21]
    M_list = GHZ_fid(kappas, 'Z')
    plt.plot(kappas, M_list)
    plt.xscale("log")
    plt.show()
