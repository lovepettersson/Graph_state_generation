import numpy as np
import random
from IndirectMeasurements.error_model_operators import spin_flip_op, rot_uni_arb, switch, optical_collapse_operators,\
    excite_collapse_operators
from scipy.optimize import minimize
import matplotlib.pyplot as plt



identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex64)

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


def gen_photon_VQE(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, angle_shift):
    psi = excite_op(psi, numb_photons)
    psi = decay_op(psi, "early", I, C, numb_photons)
    psi = rot_uni_arb(7, psi, 0.00000000000000001, numb_photons, C1, C2, 0, 0)
    psi = excite_op(psi, numb_photons)
    psi = decay_op(psi, "late", I, C, numb_photons)
    psi = rot_uni_arb(7 + angle_shift, psi, kappa, numb_photons, C1, C2, deltaOH, 0)
    # psi = np.matmul(H, psi)
    # v, psi = switch(psi, numb_photons)
    return psi


def three_star_VQE(I, kappa, C, numb_photons, angle_shift0, angle_shift1):
    C1, C2 = spin_flip_op(kappa)
    spin = np.array([1, 0, 0, 0])
    photon = np.array([1, 0, 0, 0])
    psi_init = np.kron(spin, photon)
    psi_sim = 0
    for i in range(numb_photons):
        if i == 0:
            C2 = np.kron(C2, identity)
            C1 = np.kron(C1, identity)
        else:
            C1 = np.kron(C1, identity)
            C2 = np.kron(C2, identity)
            psi_init = np.kron(psi_init, photon)

    # deltaOH = 0
    deltaOH = np.random.normal(0, np.sqrt(2) / 23.2)
    for i in range(100):
        psi = psi_init
        psi = rot_uni_arb(3.5 + angle_shift0, psi, kappa, numb_photons, C1, C2, deltaOH, np.pi / 2)
        psi = gen_photon_VQE(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, angle_shift1)
        den = np.outer(psi, np.conjugate(psi).transpose())
        psi_sim += den / 100
    # psi = gen_photon_VQE(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, angle_shift2)
    # psi = gen_photon_VQE(psi, kappa, C1, C2, I, C, deltaOH, H, numb_photons, angle_shift3)
    post_select = post_select_op(numb_photons)
    psi_sim = np.matmul(np.matmul(post_select, psi_sim), post_select)
    norm = np.trace(psi_sim)
    psi_sim = psi_sim / norm

    # return psi
    return psi_sim

def hamiltonian(coeff1, coeff2, lamb):
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

    XX = -np.kron(pauliX_spin, pauliX_photon)
    ZZ = -np.kron(pauliZ_spin, pauliZ_photon)
    peturb1 = coeff1 * np.kron(pauliZ_spin, identity)
    peturb2 = coeff2 * np.kron(identity, pauliZ_photon)
    H = XX + ZZ + lamb * (peturb1 + peturb2)
    return H

def objective(v, *args):
    I = 0.947
    C = 14.7
    kappa = 0.021
    lamb, H = args
    numb_photons = 1
    x, y = v
    density_matrix = three_star_VQE(I, C, kappa, numb_photons, x, y)
    return np.real(np.trace(np.matmul(H, density_matrix)))



#args=(D_neg, D, C)
def optimizer(lambdas, coeff1, coeff2):
    diag = []
    vqe = []
    parameter1 = []
    parameter2 = []
    init1 = 1
    init2 = 0
    for lamb in lambdas:
        H = hamiltonian(coeff1, coeff2, lamb)
        result = minimize(objective, [init1, init2], args=(lamb, H), method='nelder-mead', options={"maxiter": 1000})
        # summarize the result
        print('Status : %s' % result['message'])
        print('Total Evaluations: %d' % result['nfev'])
        # evaluate solution
        solution = result['x']
        evaluation = objective(solution)
        vqe.append(evaluation)
        angle1, angle2 = solution
        init1, init2 = angle1, angle2
        parameter1.append(angle1)
        parameter2.append(angle2)
        print('Solution: f(%s) = %.5f' % (solution, evaluation))
        diag.append(min(np.linalg.eigvals(H)))


    error = [abs((diag[i] - vqe[i]) / diag[i]) for i in range(15)]
    plt.plot(lambdas, vqe, 'bo', label="VQE")
    plt.plot(lambdas, diag, 'r', label="mode 0")
    plt.xlabel("$\lambda$")
    plt.xscale("log")
    plt.ylabel("E")
    plt.legend()
    plt.show()

    plt.plot(lambdas, error, 'bo', label="error")
    plt.xlabel("$\lambda$")
    plt.ylabel("Error (percentage)")
    plt.legend()
    plt.show()

    plt.plot(lambdas, parameter1, label="1")
    plt.plot(lambdas, parameter2, label="2")
    plt.xlabel("$\lambda$")
    plt.ylabel("$\delta_{T_p} [ns]$")
    plt.legend()
    plt.show()

    print("Average error {}".format(np.mean(error)))

if __name__ == "__main__":
    # Running for the even case, i.e. both perturbation parameters of equal strength.
    coeff1 = 1
    coeff2 = 1
    lambdas = np.linspace(0, 5, 15)
    optimizer(lambdas, coeff1, coeff2)
