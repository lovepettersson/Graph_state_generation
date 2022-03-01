import numpy as np
from grover_deutsch_graphs import gen_box, gen_line, gen_box_ideal



def measurement(ang1, ang2):
    # SPIN = qudit_1, PHOTON_1 = qudit_2, etc....
    qudit_1 = np.array([1 / np.sqrt(2), np.exp(1j*ang1) / np.sqrt(2), 0, 0], dtype=np.complex128)
    qudit_2 = np.array([0, 1 / np.sqrt(2), np.exp(1j*ang2) / np.sqrt(2), 0], dtype=np.complex128)
    qudit_1_minus = np.array([1 / np.sqrt(2), -np.exp(1j * ang1) / np.sqrt(2), 0, 0], dtype=np.complex128)
    qudit_2_minus = np.array([0, 1 / np.sqrt(2), -np.exp(1j * ang2) / np.sqrt(2), 0], dtype=np.complex128)
    first_list = [qudit_1, qudit_1_minus]
    qudit_3 = np.array([0, 0, 1, 0], dtype=np.complex128)
    qudit_4 = np.array([0, 0, 1, 0], dtype=np.complex128)
    qudit_3_zero = np.array([0, 1, 0, 0], dtype=np.complex128)
    qudit_4_zero = np.array([0, 1, 0, 0], dtype=np.complex128)

    Z_H = np.array([[1, 0, 0, 0],
                         [0, 1 / (2 ** (1 / 2)), 1 / (2 ** (1 / 2)), 0],
                         [0, -1 / (2 ** (1 / 2)), 1 / (2 ** (1 / 2)), 0],
                         [0, 0, 0, 1],
                         ])


    qudit_3 = np.matmul(Z_H, qudit_3)
    qudit_4 = np.matmul(Z_H, qudit_4)
    qudit_3_zero = np.matmul(Z_H, qudit_3_zero)
    qudit_4_zero = np.matmul(Z_H, qudit_4_zero)

    # s_2=0, s_4=0, i.e no pauli errors
    base11 = np.kron(np.kron(np.kron(qudit_1, qudit_2), qudit_3), qudit_4)
    base00 = np.kron(np.kron(np.kron(qudit_1, qudit_2), qudit_3_zero), qudit_4_zero)
    base01 = np.kron(np.kron(np.kron(qudit_1, qudit_2), qudit_3_zero), qudit_4)
    base10 = np.kron(np.kron(np.kron(qudit_1, qudit_2), qudit_3), qudit_4_zero)

    basis0 = [base00, base01, base10, base11]
    return basis0


def measurement_deutsch_constant():
    # SPIN = qudit_1, PHOTON_1 = qudit_2, etc....
    qudit_10 = np.array([1, 0, 0, 0], dtype=np.complex128)
    qudit_20 = np.array([0, 0, 1, 0], dtype=np.complex128)
    qudit_30 = np.array([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0], dtype=np.complex128)
    qudit_40 = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=np.complex128)

    qudit_11 = np.array([0, 1, 0, 0], dtype=np.complex128)
    qudit_21 = np.array([0, 1, 0, 0], dtype=np.complex128)
    qudit_31 = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=np.complex128)
    qudit_41 = np.array([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0], dtype=np.complex128)

    #qudit_1 = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128) ## SPIN
    #qudit_2 = np.array([1, 0], dtype=np.complex128) ## 1
    #qudit_4 = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=np.complex128) ### 2
    #qudit_3 = np.array([0, 1], dtype=np.complex128) #### 3
    base01 = np.kron(np.kron(np.kron(qudit_10, qudit_20), qudit_30), qudit_40)
    base00 = np.kron(np.kron(np.kron(qudit_10, qudit_21), qudit_30), qudit_40)
    base11 = np.kron(np.kron(np.kron(qudit_10, qudit_20), qudit_30), qudit_41)
    base10 = np.kron(np.kron(np.kron(qudit_10, qudit_21), qudit_30), qudit_41)

    return base00, base01, base10, base11


def measurement_deutsch_balanced():
    # SPIN = qudit_1, PHOTON_1 = qudit_2, etc....
    qudit_10 = np.array([1 / np.sqrt(2), -1j / np.sqrt(2), 0, 0], dtype=np.complex128) # qubit 2
    qudit_20 = np.array([0, 1, 0, 0], dtype=np.complex128) # qubit 3
    qudit_30 = np.array([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0], dtype=np.complex128) # qubit 4
    qudit_40 = np.array([0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0], dtype=np.complex128) # qubit 1

    qudit_11 = np.array([1 / np.sqrt(2), 1j / np.sqrt(2), 0, 0], dtype=np.complex128)  # qubit 2
    qudit_21 = np.array([0, 0, 1, 0], dtype=np.complex128)  # qubit 3
    qudit_31 = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=np.complex128)  # qubit 4
    qudit_41 = np.array([0, 1 / np.sqrt(2), -1j / np.sqrt(2), 0], dtype=np.complex128)  # qubit 1

    ## change to qudit11 and qudit31 for s_2 = 1 and s_4 = 1 respectivley
    base00 = np.kron(np.kron(np.kron(qudit_11, qudit_20), qudit_31), qudit_40)
    base01 = np.kron(np.kron(np.kron(qudit_11, qudit_21), qudit_31), qudit_40)
    base10 = np.kron(np.kron(np.kron(qudit_11, qudit_20), qudit_31), qudit_41)
    base11 = np.kron(np.kron(np.kron(qudit_11, qudit_21), qudit_31), qudit_41)

    return base00, base01, base10, base11


def run_deutsch(density_matrix, t):
    if t == "balanced":
        base = measurement_deutsch_balanced()
    else:
        base = measurement_deutsch_constant()

    fid00 = np.matmul(np.matmul(np.conjugate(base[0]).transpose(), density_matrix), base[0])
    fid01 = np.matmul(np.matmul(np.conjugate(base[1]).transpose(), density_matrix), base[1])
    fid10 = np.matmul(np.matmul(np.conjugate(base[2]).transpose(), density_matrix), base[2])
    fid11 = np.matmul(np.matmul(np.conjugate(base[3]).transpose(), density_matrix), base[3])
    norm = fid11 + fid10 + fid01 + fid00
    fid00 = fid00 / norm
    fid01 = fid01 / norm
    fid10 = fid10 / norm
    fid11 = fid11 / norm
    return fid00, fid01, fid10, fid11




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fid_list = []
    for j in range(20):
        density_matrix = gen_box(0.947, 0.021, 23.2, 3)
        ideal = gen_box_ideal(1, 0.000000000000001, 9999999999, 3)
        fid = np.matmul(np.matmul(np.conjugate(ideal).transpose(), density_matrix), ideal)
        fid_list.append(fid)
    print(np.mean(fid_list))
    print(np.std(fid_list))

    '''
    collection = []
    fid00 = []
    fid01 = []
    fid10 = []
    fid11 = []

    fid00_00 = []
    fid01_00 = []
    fid10_00 = []
    fid11_00 = []

    fid00_11 = []
    fid01_11 = []
    fid10_11 = []
    fid11_11 = []

    fid00_10 = []
    fid01_10 = []
    fid10_10 = []
    fid11_10 = []
    for j in range(10):
        density_matrix = gen_box(0.947, 0.021, 23.2, 3)
        basis_states = measurement(0, np.pi)
        fd00 = np.matmul(np.matmul(np.conjugate(basis_states[0]).transpose(), density_matrix), basis_states[0])
        fd01 = np.matmul(np.matmul(np.conjugate(basis_states[1]).transpose(), density_matrix), basis_states[1])
        fd10 = np.matmul(np.matmul(np.conjugate(basis_states[2]).transpose(), density_matrix), basis_states[2])
        fd11 = np.matmul(np.matmul(np.conjugate(basis_states[3]).transpose(), density_matrix), basis_states[3])
        norm = fd00 + fd01 + fd10 + fd11
        fid00.append(fd00 / norm)
        fid01.append(fd01 / norm)
        fid10.append(fd10 / norm)
        fid11.append(fd11 / norm)

        basis_states = measurement(0, 0)
        fd00 = np.matmul(np.matmul(np.conjugate(basis_states[0]).transpose(), density_matrix), basis_states[0])
        fd01 = np.matmul(np.matmul(np.conjugate(basis_states[1]).transpose(), density_matrix), basis_states[1])
        fd10 = np.matmul(np.matmul(np.conjugate(basis_states[2]).transpose(), density_matrix), basis_states[2])
        fd11 = np.matmul(np.matmul(np.conjugate(basis_states[3]).transpose(), density_matrix), basis_states[3])
        norm = fd00 + fd01 + fd10 + fd11
        fid00_00.append(fd00 / norm)
        fid01_00.append(fd01 / norm)
        fid10_00.append(fd10 / norm)
        fid11_00.append(fd11 / norm)

        basis_states = measurement(np.pi, np.pi)
        fd00 = np.matmul(np.matmul(np.conjugate(basis_states[0]).transpose(), density_matrix), basis_states[0])
        fd01 = np.matmul(np.matmul(np.conjugate(basis_states[1]).transpose(), density_matrix), basis_states[1])
        fd10 = np.matmul(np.matmul(np.conjugate(basis_states[2]).transpose(), density_matrix), basis_states[2])
        fd11 = np.matmul(np.matmul(np.conjugate(basis_states[3]).transpose(), density_matrix), basis_states[3])
        norm = fd00 + fd01 + fd10 + fd11
        fid00_11.append(fd00 / norm)
        fid01_11.append(fd01 / norm)
        fid10_11.append(fd10 / norm)
        fid11_11.append(fd11 / norm)

        basis_states = measurement(np.pi, 0)
        fd00 = np.matmul(np.matmul(np.conjugate(basis_states[0]).transpose(), density_matrix), basis_states[0])
        fd01 = np.matmul(np.matmul(np.conjugate(basis_states[1]).transpose(), density_matrix), basis_states[1])
        fd10 = np.matmul(np.matmul(np.conjugate(basis_states[2]).transpose(), density_matrix), basis_states[2])
        fd11 = np.matmul(np.matmul(np.conjugate(basis_states[3]).transpose(), density_matrix), basis_states[3])
        norm = fd00 + fd01 + fd10 + fd11
        fid00_10.append(fd00 / norm)
        fid01_10.append(fd01 / norm)
        fid10_10.append(fd10 / norm)
        fid11_10.append(fd11 / norm)


    print("mean 00 prob is {}, mean 01 prob is {}, mean 10 prob is {} and mean 11 prob is {}".format(np.mean(fid00),\
                                                                                                     np.mean(fid01),\
                                                                                                     np.mean(fid10),\
                                                                                                     np.mean(fid11)))

    print("std 00 prob is {}, std 01 prob is {}, std 10 prob is {} and std 11 prob is {}".format(np.std(fid00), \
                                                                                                     np.std(fid01), \
                                                                                                     np.std(fid10), \
                                                                                                     np.std(fid11)))
    ### fid = <ideal|p|ideal> = probability!
    x = (abs(np.mean(fid00)), abs(np.mean(fid01)), abs(np.mean(fid10)), abs(np.mean((fid11))))
    error = (np.std(fid00), np.std(fid01), np.std(fid10), np.std(fid11))
    print(x)
    y = ("|00>", "|01>", "|10>", "|11>")
    plt.bar(y, x, 0.4, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.ylabel('Probability')
    plt.title(r'$\alpha = 0, \beta = \pi$')
    #plt.legend(['\u03B1 = 0'], ['\u03B2 = 0'])

    plt.show()

    x = (abs(np.mean(fid00_00)), abs(np.mean(fid01_00)), abs(np.mean(fid10_00)), abs(np.mean((fid11_00))))
    error = (np.std(fid00_00), np.std(fid01_00), np.std(fid10_00), np.std(fid11_00))
    print(x)
    y = ("|00>", "|01>", "|10>", "|11>")
    plt.bar(y, x, 0.4, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.ylabel('Probability')
    plt.title(r'$\alpha = 0, \beta = 0$')
    # plt.legend(['\u03B1 = 0'], ['\u03B2 = 0'])

    plt.show()

    x = (abs(np.mean(fid00_10)), abs(np.mean(fid01_10)), abs(np.mean(fid10_10)), abs(np.mean((fid11_10))))
    error = (np.std(fid00_10), np.std(fid01_10), np.std(fid10_10), np.std(fid11_10))
    print(x)
    y = ("|00>", "|01>", "|10>", "|11>")
    plt.bar(y, x, 0.4, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.ylabel('Probability')
    plt.title(r'$\alpha = \pi, \beta = 0$')
    # plt.legend(['\u03B1 = 0'], ['\u03B2 = 0'])

    plt.show()

    x = (abs(np.mean(fid00_11)), abs(np.mean(fid01_11)), abs(np.mean(fid10_11)), abs(np.mean((fid11_11))))
    error = (np.std(fid00_11), np.std(fid01_11), np.std(fid10_11), np.std(fid11_11))
    print(x)
    y = ("|00>", "|01>", "|10>", "|11>")
    plt.bar(y, x, 0.4, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.ylabel('Probability')
    plt.title(r'$\alpha = \pi, \beta = \pi$')
    # plt.legend(['\u03B1 = 0'], ['\u03B2 = 0'])

    plt.show()
    '''
    '''
    fid00_constant = []
    fid01_constant = []
    fid10_constant = []
    fid11_constant = []

    fid00_balanced = []
    fid01_balanced = []
    fid10_balanced = []
    fid11_balanced = []
    for j in range(10):
        density_matrix = gen_line(0.947, 0.0021, 23.2, 3)
        # density_matrix = gen_line(1, 0.000000000021, 23333333333333.2, 3)
        fid00, fid01, fid10, fid11 = run_deutsch(density_matrix, "balanced")
        fid00_balanced.append(fid00)
        fid01_balanced.append(fid01)
        fid10_balanced.append(fid10)
        fid11_balanced.append(fid11)

        fid00, fid01, fid10, fid11 = run_deutsch(density_matrix, "constant")
        fid00_constant.append(fid00)
        fid01_constant.append(fid01)
        fid10_constant.append(fid10)
        fid11_constant.append(fid11)

    x = (abs(np.mean(fid00_balanced)), abs(np.mean(fid01_balanced)), abs(np.mean(fid10_balanced)), abs(np.mean((fid11_balanced))))
    error = (np.std(fid00_balanced), np.std(fid01_balanced), np.std(fid10_balanced), np.std(fid11_balanced))
    print(x)
    y = ("|11>", "|01>", "|10>", "|00>")
    plt.bar(y, x, 0.4, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.ylabel('Probability')
    plt.title(r'Balanced function with $s_2 = 0, s_4 = 0$')
    plt.show()

    x = (abs(np.mean(fid00_constant)), abs(np.mean(fid01_constant)), abs(np.mean(fid10_constant)),
         abs(np.mean((fid11_constant))))
    error = (np.std(fid00_constant), np.std(fid01_constant), np.std(fid10_constant), np.std(fid11_constant))
    print(x)
    y = ("|11>", "|01>", "|10>", "|00>")
    plt.bar(y, x, 0.4, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.ylabel('Probability')
    plt.title(r'Constant function with $s_2 = 0, s_4 = 0$')
    plt.show()
    '''