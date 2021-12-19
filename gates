import numpy as np

identity = np.array([[1, 0],
                     [0, 1]])

exc_matrix = np.array([[1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1],
                               ])

exc_error_matrix = np.array([[1 - 10**(-2), 0, 0, 10**(-2)],
                               [0, 0, 1, 0],
                               [0, 1, 0, 0],
                               [-10**(-2), 0, 0, 1 - 10**(-2)],
                               ])

X_half_matrix = np.array([[1 / np.sqrt(2), -1j/np.sqrt(2), 0, 0],
                               [-1j/np.sqrt(2), 1 / np.sqrt(2), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1],
                               ])
X_full_matrix = np.array([[0, 1, 0, 0],
                               [1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1],
                               ])
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


def switch_density(density_matrix, new_v_state, new_u_state, size):
    """ switching order of photon 3 and photon 1, or and other photon with one"""
    new_density_matrix = np.zeros((size, size), dtype=np.complex128)
    for row in range(len(density_matrix[0])):
        for column in range(len(density_matrix[0])):
            new_density_matrix[row][column] = density_matrix[new_u_state[row]][new_v_state[column]]
    return new_density_matrix



def CNOT():
    cnot_SA = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    cnot_PA = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    return cnot_SA, cnot_PA











