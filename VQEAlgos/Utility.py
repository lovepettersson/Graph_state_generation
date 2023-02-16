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
    identity_part = np.array([[1 / 2, 0, 0, 0],
                              [0, 1 * np.cos(theta), 0, 0],
                              [0, 0, 1 * np.cos(theta), 0],
                              [0, 0, 0, 1 / 2]])

    x_part = np.array([[1 / 2, 0, 0, 0],
                       [0, 0, 1 * (-1j) * np.sin(theta) * np.cos(phi), 0],
                       [0, 1 * (-1j) * np.sin(theta) * np.cos(phi), 0, 0],
                       [0, 0, 0, 1 / 2]])

    y_part = np.array([[1 / 2, 0, 0, 0],
                       [0, 0, -1j * (-1j) * np.sin(theta) * np.sin(phi), 0],
                       [0, 1j * (-1j) * np.sin(theta) * np.sin(phi), 0, 0],
                       [0, 0, 0, 1 / 2]])

    z_part = np.array([[1 / 2, 0, 0, 0],
                       [0, 1 * (-1j) * np.sin(theta), 0, 0],
                       [0, 0, -1 * (-1j) * np.sin(theta), 0],
                       [0, 0, 0, 1 / 2]]
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




################### OLD UNITARY DECAY OPS ##########################

class QuditEarlySPhoton():
    # beta to be 1 in ideal case
    def __init__(self, beta_par, beta_ort):
        super(QuditEarlySPhoton, self)
        self.beta_par = beta_par
        self.beta_ort = beta_ort

    def unitary(self):
        lamb = (1 - (self.beta_par ** 2) - (self.beta_ort) ** 2) ** (1 / 2)
        photo = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, lamb, 0, 0, 0, 0, 0, self.beta_ort, 0, 0, 0, self.beta_par, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, lamb, 0, -self.beta_par, 0, 0, 0, self.beta_ort, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, -self.beta_ort, 0, 0, 0, self.beta_par, 0, lamb, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, -self.beta_par, 0, 0, 0, -self.beta_ort, 0, 0, 0, 0, 0, lamb, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return photo

class QuditLateSPhoton():
    # beta to be 1 in ideal case
    def __init__(self, beta_par, beta_ort):
        super(QuditLateSPhoton, self)
        self.beta_par = beta_par
        self.beta_ort = beta_ort

    def unitary(self):

        lamb = (1 - (self.beta_par ** 2) - (self.beta_ort) ** 2) ** (1 / 2)
        photo = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, lamb, 0, 0, 0, 0, 0, 0, self.beta_ort, 0, 0, 0, self.beta_par, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, lamb, 0, 0, 0, 0, 0, 0, self.beta_ort, 0, 0, 0, self.beta_par, 0], # |32> to |03>!
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, lamb, 0, 0, -self.beta_par, 0, 0, 0, self.beta_ort, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, lamb, 0, 0, -self.beta_par, 0, 0, 0, self.beta_ort, 0],
                          [0, -self.beta_ort, 0, 0, 0, self.beta_par, 0, 0, lamb, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, -self.beta_ort, 0, 0, 0, self.beta_par, 0, 0, lamb, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, -self.beta_par, 0, 0, 0, -self.beta_ort, 0, 0, 0, 0, 0, 0, lamb, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, -self.beta_par, 0, 0, 0, -self.beta_ort, 0, 0, 0, 0, 0, 0, lamb, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        return photo

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    pauliX_spin = np.array([[0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=float)

    pauliX_photon = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]], dtype=float)

    pauliZ_spin = np.array([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=float)

    pauliZ_photon = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]], dtype=float)

    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)

    diag = []
    XX = -np.kron(pauliX_spin, pauliX_photon)
    ZZ = -np.kron(pauliZ_spin, pauliZ_photon)
    peturb1 = np.kron(pauliZ_spin, identity)
    peturb2 = np.kron(identity, pauliZ_photon)
    H = XX + ZZ +  (peturb1)
    print(min(np.linalg.eigvals(H)))


    plot_val = [0.17794569810622757, 0.07531379043155698, 0.018922409608756484,
                 0.012396828316419707, 0.01114204721857939, 0.005435991998503537, 0.00482191195572537]
    plot_error = [0.14501802739480355, 0.06635921888908239, 0.018083886648993866,
                  0.014211935661367, 0.016096284263892546, 0.0066688157848565615, 0.003586481344022354]

    plot_val_2 = [0.13234569271676216, 0.05061631596838743, 0.038894481225045435,
                 0.03400000647061462, 0.03160427391473754, 0.012849718324244844, 0.003985392935622053]
    plot_error_2 = [0.09126075065339888, 0.034328687086848234, 0.045669484053939315,
                   0.018424476388900178, 0.03431045436712623, 0.009695424067126413, 0.003032358215103618]

    plot_val_3 = [0.2220567322709393, 0.08943639070052735, 0.0575476958962742,
                 0.02161948450900253, 0.011228288063374049, 0.010086483878608369,
                 0.008889054966923902]
    plot_error_3 = [0.1325087352233709, 0.061326836073823596, 0.036929404320195876,
                   0.01617049061860478, 0.007439317786089233, 0.012997620405826867,
                   0.014980446187660392]
    T2 = [13.2, 23.2, 33.2, 43.2, 53.2, 63.2, 73.2]
    line = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    plt.errorbar(T2, plot_val, yerr=plot_error, marker="o",label="VQE")
    plt.plot(T2, line, "k:", label="1%")
    plt.axvline(x=abs((23.2)), linestyle='--', color='r')
    plt.xlabel("$T_2^*(ns)$")
    plt.ylabel("\u0394E/E (%)")
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(ncols=3, sharey=True)
    axes[0].errorbar(T2, plot_val, yerr=plot_error, marker="o",label="VQE")
    axes[0].plot(T2, line, "k:", label="1%")
    axes[0].axvline(x=abs((23.2)), linestyle='--', color='r')
    axes[1].errorbar(T2, plot_val_2, yerr=plot_error_2, marker="o", label="VQE")
    axes[1].plot(T2, line, "k:", label="1%")
    axes[1].axvline(x=abs((23.2)), linestyle='--', color='r')
    axes[2].errorbar(T2, plot_val_3, yerr=plot_error_3, marker="o", label="VQE")
    axes[2].plot(T2, line, "k:", label="1%")
    axes[2].axvline(x=abs((23.2)), linestyle='--', color='r')


    axes[0].title.set_text('$\lambda_1 = 1, \lambda_2 = 0$')
    axes[1].title.set_text('$\lambda_1 = \lambda_2 = 1$')
    axes[2].title.set_text('$\lambda_1 = -0.3, \lambda_2 = 0.6$')
    # axes[1][1].title.set_text('Fourth Plot')
    axes[0].set_ylabel("\u0394E/E (%)")
    axes[0].set_xlabel("$T_2^* (ns)$")
    axes[1].set_xlabel("$T_2^* (ns)$")
    axes[2].set_xlabel("$T_2^* (ns)$")


    fig.tight_layout()

    plt.show()

    data = np.random.rand(15, 15)
    fig, axes = plt.subplots(3, 2)

    axes[0, 0].imshow(data)
    axes[1, 0].imshow(data)
    axes[2, 0].imshow(data)

    axes[0, 1].imshow(data)
    axes[1, 1].imshow(data)
    axes[2, 1].imshow(data)

    plt.setp(axes, xticks=[], yticks=[])
    plt.show()