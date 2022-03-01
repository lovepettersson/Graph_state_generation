import numpy as np
from tqdm import tqdm
from graph_generator import ghz
from error_model_operators import measurement_rot_unitary



def projector_GHZ(theta, axis):
    U = measurement_rot_unitary(theta, axis, 0)
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex128)


    photon_X_up = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=np.complex128)
    photon_X_down = np.array([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0], dtype=np.complex128)
    photon_Z_up = np.array([0, 1, 0, 0], dtype=np.complex128)
    photon_Z_down = np.array([0, 0, 1, 0], dtype=np.complex128)
    photon_Y_up = np.array([0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0], dtype=np.complex128)
    photon_Y_down = np.array([0, 1 / np.sqrt(2), -1j / np.sqrt(2), 0], dtype=np.complex128)
    spin_Z_up = np.array([1, 0, 0, 0], dtype=np.complex128)
    spin_X_up = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], dtype=np.complex128)
    spin_Y_up = np.array([1 / np.sqrt(2), 1j / np.sqrt(2), 0, 0], dtype=np.complex128)

    photon_X_up = np.outer(photon_X_up, photon_X_up.transpose())
    photon_X_up = np.matmul(np.matmul(U, photon_X_up), np.conjugate(U).transpose())
    photon_X_down = np.outer(photon_X_down, photon_X_down.transpose())
    photon_X_down = np.matmul(np.matmul(U, photon_X_down), np.conjugate(U).transpose())
    spin_X_up = np.outer(spin_X_up, spin_X_up.transpose())
    spin_Z_up = np.outer(spin_Z_up, spin_Z_up.transpose())
    spin_Y_up = np.outer(spin_Y_up, np.conjugate(spin_Y_up).transpose())
    photon_Z_up = np.outer(photon_Z_up, np.conjugate(photon_Z_up).transpose())
    photon_Z_up = np.matmul(np.matmul(U, photon_Z_up),np.conjugate(U).transpose())
    photon_Z_down = np.outer(photon_Z_down, np.conjugate(photon_Z_down).transpose())
    photon_Z_down = np.matmul(np.matmul(U, photon_Z_down), np.conjugate(U).transpose())
    photon_Y_up = np.outer(photon_Y_up, np.conjugate(photon_Y_up).transpose())
    photon_Y_up = np.matmul(np.matmul(U, photon_Y_up), np.conjugate(U).transpose())
    photon_Y_down = np.outer(photon_Y_down, np.conjugate(photon_Y_down).transpose())
    photon_Y_down = np.matmul(np.matmul(U, photon_Y_down), np.conjugate(U).transpose())

    X_down_photon = np.kron(identity, photon_X_down)
    X_up_photon = np.kron(identity, photon_X_up)
    Z_up_photon = np.kron(identity, photon_Z_up)
    Z_down_photon = np.kron(identity, photon_Z_down)
    Y_up_photon = np.kron(identity, photon_Y_up)
    Y_down_photon = np.kron(identity, photon_Y_down)
    X_up_spin = np.kron(spin_X_up, identity)
    Z_up_spin = np.kron(spin_Z_up, identity)
    Y_up_spin = np.kron(spin_Y_up, identity)


    return X_up_spin, X_up_photon, X_down_photon, Y_up_spin, Z_up_spin, Z_up_photon, Z_down_photon, Y_up_photon, Y_down_photon


def prob_den(operator, den_matrix):
    projection = np.matmul(operator, np.matmul(den_matrix, np.conjugate(operator).transpose()))
    prob = np.trace(projection)
    projection = projection / prob
    return prob, projection



def GHZ_fid(kappas, Logical_M, T2, theta, axis, I, loss):
    numb_photons = 1
    X_up_spin, X_up_photon, X_down_photon, Y_up_spin, Z_up_spin, Z_up_photon, Z_down_photon,\
    Y_up_photon, Y_down_photon = projector_GHZ(theta, axis)
    xval = []
    the_list = [i for i in range(10)]
    for i in tqdm(the_list):
        kappa = kappas
        den_matrix = ghz(I, kappa, T2)
        if Logical_M == "X":
            P, den_matrix = prob_den(X_up_spin, den_matrix)
            P_up, den_matrix0 = prob_den(X_up_photon, den_matrix)
            P_down, den_matrix1 = prob_den(X_down_photon, den_matrix)
            P = P_up / (P_up + P_down)

        elif Logical_M == "Z":
            P, den_matrix = prob_den(Z_up_spin, den_matrix)
            # P, den_matrix = prob_den(Z_up_photon, den_matrix)
            P_up, den_matrix0 = prob_den(Z_up_photon, den_matrix)
            P_down, den_matrix1 = prob_den(Z_down_photon, den_matrix)
            P = P_up / (P_up + P_down)

        else:
            P, den_matrix = prob_den(Y_up_spin, den_matrix)
            # P, den_matrix = prob_den(Y_up_photon, den_matrix)
            P_up, den_matrix0 = prob_den(Y_up_photon, den_matrix)
            P_down, den_matrix1 = prob_den(Y_down_photon, den_matrix)
            P = P_down / (P_up + P_down)

        M = 0
        for j in range(20000):
            if P > np.random.uniform():
                continue
            else:
                M += 1 / 20000
        xval.append(M)
    error_mean = np.mean(xval)
    error = np.std(xval)
    return error_mean, error  # M

