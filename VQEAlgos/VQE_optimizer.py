from scipy.optimize import minimize
import numpy as np
from photon_detection import three_star_VQE, three_star_VQE_two
from sympy import*
import matplotlib.pyplot as plt
import json

def pauli_ops():
    pauliX_spin = np.array([[0, 1, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=float)

    pauliY_spin = np.array([[0, -1j, 0, 0],
                       [1j, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=complex)

    pauliX_photon = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]], dtype=float)

    pauliY_photon = np.array([[1, 0, 0, 0],
                       [0, 0, -1j, 0],
                       [0, 1j, 0, 0],
                       [0, 0, 0, 1]], dtype=complex)

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

    return pauliX_photon, pauliY_photon, pauliZ_photon, pauliX_spin,\
           pauliY_spin, pauliZ_spin, identity



def run_vqe(path):
    diag = []
    vqe = []
    vqe_error = []
    qiskit_sol = []
    i = 0
    # path = r"C:\Users\Admin\coeff_data.json"
    f = open(path)
    the_dict = json.load(f)
    pauliX_photon, pauliY_photon, pauliZ_photon, pauliX_spin, \
    pauliY_spin, pauliZ_spin, identity = pauli_ops()
    init = 3.5
    init2 = 0
    for key in the_dict.keys():
        coeff = the_dict[key]
        one, two, three, four, five = coeff[0]
        # one = abs(one)
        two = abs(two)
        three = abs(three)
        # four = abs(four)
        # five = abs(five)
        vqe_placeholder = []
        exact_placeholder = []
        for _ in range(10):
            II = one * np.kron(identity, identity)
            XX = five * np.kron(pauliX_spin, pauliX_photon)
            ZZ = four * np.kron(pauliZ_spin, pauliZ_photon)
            peturb1 = two * np.kron(pauliZ_spin, identity)
            peturb2 = three * np.kron(identity, pauliZ_photon)
            H = II + XX + ZZ + peturb1 + peturb2


            def objective_1(v):
                I = 1
                kappa = 0.0000000000000000000001
                numb_photons = 1
                # x, y, z, p = v
                x, y = v
                density_matrix = three_star_VQE(I, numb_photons, x, y, kappa, T2=23.2)
                # density_matrix = get_VQE_state(x, y)
                term1 = np.trace(np.matmul(XX, density_matrix))
                term2 = np.trace(np.matmul(ZZ, density_matrix))
                term3 = np.trace(np.matmul(peturb1, density_matrix))
                term4 = np.trace(np.matmul(peturb2, density_matrix))
                term5 = np.trace(np.matmul(II, density_matrix))
                return np.real(term1 + term2 + term3 + term4 + term5) + np.imag(term1 + term2 + term3 + term4 +term5)
                # return np.real(term1 + term2 + term4 + term5) + np.imag(term1 + term2 + term4 + term5)


            # r_min, r_max = -7.0, 7.0
            # define the bounds on the search
            # bounds = [[r_min, r_max]]
            # perform the dual annealing search
            # result = dual_annealing(objective_1, bounds)
            # result = basinhopping(objective_1, [init1])
            result = minimize(objective_1, [init, init2], method='nelder-mead', options={"maxiter":1000})
            # summarize the result
            print('Status : %s' % result['message'])
            print('Total Evaluations: %d' % result['nfev'])
            # evaluate solution
            solution = result['x']
            evaluation = objective_1(solution)
            repulsion = coeff[2]
            # vqe.append(evaluation + repulsion)
            vqe_placeholder.append(evaluation + repulsion)
            # angle1, angle2, angle3, angle4 = solution
            # angle1 = solution
            # angle2 = 0
            # init1 = ini[i]
            init, init2 = solution
            # init2 = angle2
            # init2 = angle2
            print('Solution: f(%s) = %.5f' % (solution, evaluation))
            # diag.append(min(np.linalg.eigvals(H)) + repulsion)
            qiskit_sol.append(coeff[1])
            exact_placeholder.append(min(np.linalg.eigvals(H)) + repulsion)
            i += 1

        vqe.append(np.mean(vqe_placeholder))
        diag.append(np.mean(exact_placeholder))
        vqe_error.append(np.std(vqe_placeholder))
    return vqe, diag, vqe_error, qiskit_sol

if __name__ == "__main__":
    path = r"C:\Users\Admin\coeff_data.json"
    f = open(path)
    the_dict = json.load(f)
    vqe, diag, vqe_error, qiskit_sol = run_vqe(path)
    error = [abs((diag[i]-vqe[i]) / diag[i]) for i in range(10)]
    distance = [float(x) for x in the_dict.keys()]
    plt.errorbar(distance, vqe, yerr=vqe_error, fmt='bo', label="VQE")
    # plt.plot(distance, diag, 'r', label="mode 0")
    plt.plot(distance, qiskit_sol, label="qiskit")
    plt.xlabel("Bond length (Ångström)")
    plt.title("Ground state energy of $H_2$")
    plt.ylabel("Eenergy (Hartee)")
    plt.legend()
    plt.show()


    print("VQE {} and DIAG {}".format(vqe, diag))
    print("Average error {}".format(np.mean(error)))
    print(np.std(error))

## NOTE:
# ----> Our model is faster by a factor of ~ 2 compared to cirq.



