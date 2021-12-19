import numpy as np
from gates import QuditEarlySPhoton, QuditLateSPhoton, switch


def spin_unitary(phi, rabi, time, deltaOH):
    # IT SHOULD BE  1/4
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



def LC_rotations_photons():
    # SO FAR THE PHOTONS GATES ARE PERFORMED WITHOUT ERRORS
    square_root = 1 / (2 ** (1 / 2))
    rot_matrix = np.array([[1, 0, 0, 0],
                           [0, (1 + 1j) * square_root, 0, 0],
                           [0, 0, (1 - 1j) * square_root, 0],
                           [0, 0, 0, 1],
                           ])
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    Z_rot_gate_1 = np.kron(np.kron(np.kron(identity, rot_matrix), identity), identity)
    Z_rot_gate_2 = np.kron(np.kron(np.kron(identity, identity), rot_matrix), identity)
    Z_rot_gate_3 = np.kron(np.kron(np.kron(identity, identity), identity), rot_matrix)

    rot_matrix = np.array([[1, 0, 0, 0],
                           [0, 1 * square_root, -1j * square_root, 0],
                           [0, -1j * square_root, 1 * square_root, 0],
                           [0, 0, 0, 1],
                           ])

    X_rot_gate_1 = np.kron(np.kron(np.kron(identity, rot_matrix), identity), identity)
    return Z_rot_gate_1, Z_rot_gate_2, Z_rot_gate_3, X_rot_gate_1


def LC_Z_spin(deltaOH):
    # THIS IS A SEPERATE FUNCTION BECAUSE WE ARE GOING TO SAMPLE THIS UNITARY DURING MC SIMULATION,
    # SPECIFICALLY DESIGN FOR DEUTSCH AND GROVER, I.E N=4.
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    Z_rot_spin = np.conjugate(spin_inverse_unitary(0, 0, np.pi / 2, 1 + deltaOH))
    Z_rot_spin = np.kron(np.kron(np.kron(Z_rot_spin, identity), identity), identity)
    return Z_rot_spin


def prob_dist(kappa):
    t_f = np.random.exponential(1/kappa)
    return t_f


def spin_flip_op(kappa):
    zero_to_one = np.array([[1, 0, 0, 0],
                            [np.sqrt(kappa), 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]) # add 1 in first row

    one_to_zero = np.array([[0, np.sqrt(kappa), 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]) # add 1 in second row

    return zero_to_one, one_to_zero



def rot_uni_arb(t_dur, state_evolved, kappa, numb_photons, C1, C2, deltaOH, psi_s):
    t_now = 0
    psi = state_evolved
    #deltaOH = np.random.normal(0, np.sqrt(2) / 23.2)
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
            pulse = spin_unitary(psi_s, np.pi / 7, t_f, 0) # Change 0 to pi/ 2 for y-rot
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
                psi = np.matmul(C2, psi) / (np.sqrt((1 - p1)))

            t_now += t_f
    return psi


def rot_uni_inv(t_dur, state_evolved, kappa, numb_photons, C1, C2, deltaOH):
    t_now = 0
    psi = state_evolved
    #deltaOH = np.random.normal(0, np.sqrt(2) / 23.2)
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


def gates_state_sec(numb_photons, beta_par, beta_dia, delta0):

    qudit = np.array([1, 0, 0, 0], dtype=np.complex64) # dtype=np.complex32
    photon = np.array([1, 0, 0, 0], dtype=np.complex64) # dtype=np.complex32
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex64) # dtype=np.complex32

    delta2 = 0.9995  # 0.955, not including Delta00


    exc_matrix = np.array([[np.cos(np.pi/2*(delta0)), 0, 0, np.sin(np.pi/2*(delta0))],
                           [0, np.cos(np.pi/2*(delta2)), np.sin(np.pi/2*(delta2)), 0],
                           [0, -np.sin(np.pi/2*(delta2)), np.cos(np.pi/2*(delta2)), 0],
                           [-np.sin(np.pi/2*(delta0)), 0, 0, np.cos(np.pi/2*(delta0))],
                           ])

    exc_matrix_perfect = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           ])

    H_matrix = np.array([[1, 0, 0, 0],
                         [0, 1 / (2 ** (1 / 2)), 1 / (2 ** (1 / 2)), 0],
                         [0, 1 / (2 ** (1 / 2)), -1 / (2 ** (1 / 2)), 0],
                         [0, 0, 0, 1],
                         ])

    H_gate_photon_1 = np.kron(identity, H_matrix)
    pi_half_y = spin_unitary(np.pi / 2, np.pi / 7, 3.5, 0)
    pi_half_pulse = spin_unitary(0, np.pi / 7, 3.5, 0) # Change 0 to pi/ 2 for y-rot
    pi_half_pulse_inv = spin_inverse_unitary(np.pi / 2, np.pi / 7, 3.5, 0) # Change 0 to pi/ 2 for y-rot
    pi_pulse = spin_unitary(0, np.pi / 7, 7, 0) # Change 0 to pi/ 2 for y-rot
    pi_half_y_gate = np.kron(pi_half_y, identity)
    pi_half_gate = np.kron(pi_half_pulse, identity)
    pi_half_inv_gate = np.kron(pi_half_pulse_inv, identity)
    pi_gate = np.kron(pi_pulse, identity)

    early_gate = QuditEarlySPhoton(beta_par, beta_dia).unitary()
    late_gate = QuditLateSPhoton(beta_par, beta_dia).unitary()
    early_gate_perfect = QuditEarlySPhoton(1, 0).unitary()
    late_gate_perfect = QuditLateSPhoton(1, 0).unitary()

    ex_perfect = np.kron(exc_matrix_perfect, identity)
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
        ex_perfect = np.kron(ex_perfect, identity)
        early_gate = np.kron(early_gate, identity)
        late_gate = np.kron(late_gate, identity)
        C1 = np.kron(C1, identity)
        C2 = np.kron(C2, identity)
        pi_gate = np.kron(pi_gate, identity)
        pi_half_gate = np.kron(pi_half_gate, identity)
        pi_half_inv_gate = np.kron(pi_half_inv_gate, identity)
        pi_half_y_gate = np.kron(pi_half_y_gate, identity)
        early_gate_perfect = np.kron(early_gate_perfect, identity)
        late_gate_perfect = np.kron(late_gate_perfect, identity)
        H_gate_photon_1 = np.kron(H_gate_photon_1, identity)

    #state = np.matmul(pi_half_y_gate, state) # HERE IS THE INTIAL ROT FOR PERFECT STATE
    return ex, ex_perfect, early_gate, late_gate, state, C1, C2, pi_half_gate, pi_gate,\
           early_gate_perfect, late_gate_perfect, H_gate_photon_1, pi_half_y_gate

def gen_LC_X_photon(numb_photons, kappa,  ex, early_gate, late_gate, state, C1, C2, deltaOH):
    #rot_state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH)
    rot_state = np.matmul(ex, state)
    rot_state = np.matmul(early_gate, rot_state)
    rot_state = rot_uni_arb(7, rot_state, kappa, numb_photons, C1, C2, deltaOH)
    rot_state = np.matmul(ex, rot_state)
    rot_state = np.matmul(late_gate, rot_state)
    rot_state = rot_uni_inv(3.5, rot_state, kappa, numb_photons, C1, C2, deltaOH)
    return rot_state


def gen_new_photon(numb_photons, kappa,  ex, early_gate, late_gate, state, C1, C2, deltaOH, H):
    #rot_state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH)
    rot_state = np.matmul(ex, state)
    rot_state = np.matmul(early_gate, rot_state)
    rot_state = rot_uni_arb(7, rot_state, kappa, numb_photons, C1, C2, deltaOH, 0)
    rot_state = np.matmul(ex, rot_state)
    rot_state = np.matmul(late_gate, rot_state)
    rot_state = rot_uni_arb(7, rot_state, kappa, numb_photons, C1, C2, deltaOH, 0)
    rot_state = np.matmul(H, rot_state)
    return rot_state

def gen_new_photon_y_x(numb_photons, kappa,  ex, early_gate, late_gate, state, C1, C2, deltaOH, H):
    rot_state = np.matmul(ex, state)
    rot_state = np.matmul(late_gate, rot_state)
    rot_state = rot_uni_arb(7, rot_state, kappa, numb_photons, C1, C2, deltaOH, 0)
    rot_state = np.matmul(ex, rot_state)
    rot_state = np.matmul(early_gate, rot_state)
    #rot_state = np.matmul(H, rot_state)
    return rot_state



def gen_new_perfect_LC_X_photon(ex, early_gate, late_gate, state, pi_half_gate, pi_gate, pi_half_inv_gate, H):
    #state = np.matmul(pi_half_gate, state)
    state = np.matmul(ex, state)
    state = np.matmul(early_gate, state)
    state = np.matmul(pi_gate, state)
    state = np.matmul(ex, state)
    state = np.matmul(late_gate, state)
    state = np.matmul(pi_half_inv_gate, state)
    state = np.matmul(H, state)
    return state

def gen_new_perfect_photon(ex, early_gate, late_gate, state, pi_half_gate, pi_gate, H):
    #state = np.matmul(pi_half_gate, state)
    state = np.matmul(ex, state)
    state = np.matmul(early_gate, state)
    state = np.matmul(pi_gate, state)
    state = np.matmul(ex, state)
    state = np.matmul(late_gate, state)
    state = np.matmul(pi_gate, state)
    state = np.matmul(H, state)
    return state

def gen_box(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, H, deltaOH, Z_rot_1, Z_rot_2,\
            Z_rot_3, Z_rot_S, X_rot_1):
    state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    #state = gen_LC_X_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH)
    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)

    state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH, 0)
    state = np.matmul(Z_rot_1, state)
    state = np.matmul(Z_rot_2, state)

    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)

    #state = gen_LC_X_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH)
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)

    state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH, 0)
    state = np.matmul(Z_rot_1, state)
    state = np.matmul(Z_rot_2, state)
    state = np.matmul(Z_rot_3, state)

    state = np.matmul(X_rot_1, state)
    state = np.matmul(Z_rot_3, state)
    state = np.matmul(Z_rot_S, state)


    return state


def gen_box_line(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, H, deltaOH, Z_rot_1, Z_rot_2,\
            Z_rot_3, Z_rot_S, X_rot_1):
    state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    #state = gen_LC_X_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH)
    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)

    state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH, 0)
    state = np.matmul(Z_rot_1, state)
    state = np.matmul(Z_rot_2, state)

    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)

    #state = gen_LC_X_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH)
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)

    state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH, 0)
    state = np.matmul(Z_rot_1, state)
    state = np.matmul(Z_rot_2, state)
    state = np.matmul(Z_rot_3, state)

    return state



def gen_box_line_ideal(numb_photons, ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, H,\
                   Z_rot_1, Z_rot_2, Z_rot_3, Z_rot_S, X_rot_1, pi_half_y_gate):
    state = np.matmul(pi_half_y_gate, state)
    state = gen_new_perfect_photon(ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, H)

    v, state = switch(state, numb_photons)
    #state = gen_new_perfect_LC_X_photon(ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, pi_half_inv, H)
    state = gen_new_perfect_photon(ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, H)


    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)
    state = np.matmul(pi_half_gate, state)
    state = np.matmul(Z_rot_1, state)
    state = np.matmul(Z_rot_2, state)
    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)

    #state = gen_new_perfect_LC_X_photon(ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, pi_half_inv, H)
    state = gen_new_perfect_photon(ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, H)

    v, state = switch(state, numb_photons)
    state = np.matmul(pi_half_gate, state)
    state = np.matmul(Z_rot_1, state)
    state = np.matmul(Z_rot_2, state)
    state = np.matmul(Z_rot_3, state)


    return state




def gen_triangle(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, H, deltaOH):
    square_root = 1 / np.sqrt(2)
    rot_matrix = np.array([[1, 0, 0, 0],
                           [0, (1 + 1j) * square_root, 0, 0],
                           [0, 0, (1 - 1j) * square_root, 0],
                           [0, 0, 0, 1],
                           ])
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    Z_rot_1 = np.kron(np.kron(identity, rot_matrix), identity)
    Z_rot_2 = np.kron(np.kron(identity, identity), rot_matrix)
    state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)
    state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
    v, state = switch(state, numb_photons)
    state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH, 0)
    state = np.matmul(Z_rot_1, state)
    state = np.matmul(Z_rot_2, state)
    return state


def gen_triangle_ideal(numb_photons, ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, H, pi_half_y_gate):

    square_root = 1 / np.sqrt(2)
    rot_matrix = np.array([[1, 0, 0, 0],
                           [0, (1 + 1j) * square_root, 0, 0],
                           [0, 0, (1 - 1j) * square_root, 0],
                           [0, 0, 0, 1],
                           ])
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    Z_rot_1 = np.kron(np.kron(identity, rot_matrix), identity)
    Z_rot_2 = np.kron(np.kron(identity, identity), rot_matrix)
    state = np.matmul(pi_half_y_gate, state)
    state = gen_new_perfect_photon(ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, H)
    v, state = switch(state, numb_photons)
    state = gen_new_perfect_photon(ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, H)
    v, state = switch(state, numb_photons)
    state = np.matmul(pi_half_gate, state)
    state = np.matmul(Z_rot_1, state)
    state = np.matmul(Z_rot_2, state)
    return state


def gen_box_ideal(numb_photons, ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, H,\
                   Z_rot_1, Z_rot_2, Z_rot_3, Z_rot_S, X_rot_1, pi_half_y_gate):
    state = np.matmul(pi_half_y_gate, state)
    state = gen_new_perfect_photon(ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, H)

    v, state = switch(state, numb_photons)
    #state = gen_new_perfect_LC_X_photon(ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, pi_half_inv, H)
    state = gen_new_perfect_photon(ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, H)


    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)
    state = np.matmul(pi_half_gate, state)
    state = np.matmul(Z_rot_1, state)
    state = np.matmul(Z_rot_2, state)
    v, state = switch(state, numb_photons)
    v, state = switch(state, numb_photons)

    #state = gen_new_perfect_LC_X_photon(ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, pi_half_inv, H)
    state = gen_new_perfect_photon(ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, H)

    v, state = switch(state, numb_photons)
    state = np.matmul(pi_half_gate, state)
    state = np.matmul(Z_rot_1, state)
    state = np.matmul(Z_rot_2, state)
    state = np.matmul(Z_rot_3, state)

    state = np.matmul(X_rot_1, state)
    state = np.matmul(Z_rot_3, state)
    state = np.matmul(Z_rot_S, state)

    return state

def gen_GHZ_ideal(numb_photons, ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, H):
    state = np.matmul(pi_half_gate, state)
    for i in range(numb_photons):
        state = gen_new_perfect_photon(ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, H)
        state = np.matmul(H, state)
        v, state = switch(state, numb_photons)
    return state

def gen_GHZ(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, H, deltaOH):
    state = rot_uni_arb(3.5, state, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
    for i in range(numb_photons):
        state = gen_new_photon(numb_photons, kappa, ex, early_gate, late_gate, state, C1, C2, deltaOH, H)
        state = np.matmul(H, state)
        v, state = switch(state, numb_photons)
    return state

if __name__ == "__main__":
    kappa = 0.00000000021
   #beta_par = 0.992
    #beta_ort = 0.072
    numb_photons = 2
    cross_ex = 0.1
    beta_par = 1
    beta_ort = 0
    #cross_ex = 0
    # GET MATRICES
    ex, ex_perfect, early_gate, late_gate, state, C1, C2, pi_half_gate, pi_gate,\
    early_gate_perfect, late_gate_perfect, H_gate_photon_1,\
    pi_half_inv_gate = gates_state_sec(numb_photons, beta_par, beta_ort, cross_ex)
    Z_spin_perfect = LC_Z_spin(0)
    Z_rot_gate_1, Z_rot_gate_2, Z_rot_gate_3, X_rot_gate_1 = LC_rotations_photons()
    print(state)
    # GENERATE IDEAL STATE
    #ideal = gen_GHZ_ideal(numb_photons, ex_perfect, early_gate_perfect, late_gate_perfect, state, pi_half_gate, pi_gate, H_gate_photon_1)

    #print(ideal)
