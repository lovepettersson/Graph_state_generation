import numpy as np
import json
import itertools
from itertools import combinations, permutations
import matplotlib.pyplot as plt
from automated_gen_graphs import get_it, get_five
from five_line_qubit_tom import GHZ_fid
from binary_trees import *
import seaborn as sns




def get_perm(per, length):
    """
    Called in "get_configurations".
    :param per:
    :param length:
    :return:
    """
    list_per = itertools.permutations(per[0], length)
    list_per = list(list_per)
    clean_duplicates = []
    for p0 in list_per:
        count = 0
        for p1 in list_per:
            if p0 == p1:
                count += 1
        if p0 in clean_duplicates:
            continue
        else:
            clean_duplicates.append(p0)
    return clean_duplicates

def get_configuration(numb):
    """
    Caveman way of getting all the configurations for graphs up to 7-qubits( including the input qubit).
    Here "+" referes to no error and "-" to error.
    :param numb: number of qubits
    :return: all the configurations
    """
    permutation_four = [['+++'], ['++-'], ['+--'], ['---']]
    permutation_five = [['+++-'], ['++--'], ['+---'], ['----'], ['++++']]
    permutation_six = [['++++-'], ['+++--'], ['++---'], ['+----'], ['+++++'], ['-----']]
    permutation_seven = [['+++++-'], ['++++--'], ['+++---'], ['++----'], ['+-----'], ['++++++'], ['------']]
    permutation_eigth = [['++++++-'], ['+++++--'], ['++++---'], ['+++----'], ['++-----'], ['+------'], ['+++++++'], ['-------']]
    configurations = []
    if numb == 4:
        for per in permutation_four:
            new = get_perm(per, 3)
            configurations = configurations + new

    elif numb == 5:
        for per in permutation_five:
            new = get_perm(per, 4)
            configurations = configurations + new

    elif numb == 6:
        for per in permutation_six:
            new = get_perm(per, 5)
            configurations = configurations + new

    elif numb == 7:
        for per in permutation_seven:
            new = get_perm(per, 6)
            configurations = configurations + new

    elif numb == 8:
        for per in permutation_eigth:
            new = get_perm(per, 7)
            configurations = configurations + new

    return configurations

def prob_den(operator, den_matrix):
    projection = np.matmul(operator, np.matmul(den_matrix, np.conjugate(operator).transpose()))
    prob = np.trace(projection)
    projection = projection / prob
    return prob, projection

def get_given_4_graph(numb, kappa):
    den_matrix = get_it(numb, kappa)
    return den_matrix

def get_given_5_graph(numb, kappa):
    den_matrix = get_five(numb, kappa)
    return den_matrix



def meas_projectors(B):
    """
    :param B: which basis to measure each qubit
    :return: the projectors corresponding to B. This is a list of lists, where the inner list contatins
    projector up and down.
    """
    # B for bases :D
    identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.complex128)
    Z_up = np.array([0, 1, 0, 0], dtype=np.complex128)
    Z_down = np.array([0, 0, 1, 0], dtype=np.complex128)
    X_up = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=np.complex128)
    X_down = np.array([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0], dtype=np.complex128)
    Y_up = np.array([0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0], dtype=np.complex128)
    Y_down = np.array([0, 1 / np.sqrt(2), -1j / np.sqrt(2), 0], dtype=np.complex128)

    Z_up = np.outer(Z_up, Z_up.transpose())
    Z_down = np.outer(Z_down, Z_down.transpose())
    X_up = np.outer(X_up, X_up.transpose())
    X_down = np.outer(X_down, X_down.transpose())
    Y_up = np.outer(Y_up, Y_up.transpose())
    Y_down = np.outer(Y_down, Y_down.transpose())

    Z_up_spin = np.array([1, 0, 0, 0], dtype=np.complex128)
    Z_up_spin = np.outer(Z_up_spin, Z_up_spin.transpose())
    Z_down_spin = np.array([0, 1, 0, 0], dtype=np.complex128)
    Z_down_spin = np.outer(Z_down_spin, Z_down_spin.transpose())
    X_up_spin = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], dtype=np.complex128)
    X_up_spin = np.outer(X_up_spin, X_up_spin.transpose())
    X_down_spin = np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0, 0], dtype=np.complex128)
    X_down_spin = np.outer(X_down_spin, X_down_spin.transpose())
    Y_up_spin = np.array([1 / np.sqrt(2), 1j / np.sqrt(2), 0, 0], dtype=np.complex128)
    Y_up_spin = np.outer(Y_up_spin, Y_up_spin.transpose())
    Y_down_spin = np.array([1 / np.sqrt(2), -1j / np.sqrt(2), 0, 0], dtype=np.complex128)
    Y_down_spin = np.outer(Y_down_spin, Y_down_spin.transpose())

    pattern = []
    for i in range(len(B)):
        if i == 0:
            if B[i] == 'X':
                op_up = X_up_spin
                op_dw = X_down_spin

            elif B[i] == 'Y':
                op_up = Y_up_spin
                op_dw = Y_down_spin

            else:
                op_up = Z_up_spin
                op_dw = Z_down_spin
                # ADD Y AS WELL
        else:
            op_up = identity
            op_dw = identity
        for j in range(1, len(B)):
            if B[i] == 'X':
                if i == j:
                    op_up = np.kron(op_up, X_up)
                    op_dw = np.kron(op_dw, X_down)
                else:
                    op_up = np.kron(op_up, identity)
                    op_dw = np.kron(op_dw, identity)

            elif B[i] == 'Y':
                if i == j:
                    op_up = np.kron(op_up, Y_up)
                    op_dw = np.kron(op_dw, Y_down)
                else:
                    op_up = np.kron(op_up, identity)
                    op_dw = np.kron(op_dw, identity)

            else:
                if i == j:
                    op_up = np.kron(op_up, Z_up)
                    op_dw = np.kron(op_dw, Z_down)
                else:
                    op_up = np.kron(op_up, identity)
                    op_dw = np.kron(op_dw, identity)
        pattern.append([op_up, op_dw])

    return pattern


def get_B(S, M):
    """
    :param S: list of all stabilizers
    :return: A list of measurement basis for each qubit with qubit 0 being element 0, qubit 1 is element 1 etc.
    """
    B = []
    for i in range(len(S[0])):
        if i == 0:
            if M == 'Z':
                B.append('Z')
            elif M == 'X':
                B.append('X')
            elif M == 'Y':
                B.append('Y')
        else:
            B.append('I')

    for stab in S:
        for i in range(len(stab)):
            if B[i] == 'I' and stab[i] != 'I':
                del B[i]
                B.insert(i, stab[i])
    return B



def conditional_M_S(M, S):
    if M == -1: # Currently only concerend with up in respective bases
        return 1
    else:
        return 0

def corrected_conditional(P):
    return min(P, 1 - P)

def logical_error(P_dict, S_count, monte_steps):
    logical_err = 0
    for the_key in P_dict.keys():
        P_err = P_dict[the_key] / S_count[the_key]
        P_err_corr = corrected_conditional(P_err)
        P_config = S_count[the_key] / monte_steps
        logical_err += P_config * P_err_corr
    return logical_err



def gen_prob_dict(pattern, den_matrix, configurations, numb_photons):
    prob_dict = {}
    ## FIRST PRINT THE TREE, LEFT CORRESPONDS TO DOWN AND RIGHT CORRESPONDS TO UP
    tree = Node(10)
    values = [i for i in range(1, 2**(numb_photons + 1))]
    for value in values:
        tree.insert(value)
    balanced_tree = balance_tree(inorder_traversal(tree))
    #balanced_tree.display()
    up = pattern[0][0]
    PP, den_matrix = prob_den(up, den_matrix)
    for config in configurations:
        root = 2 ** (numb_photons)
        value = root
        den_matrix_loop = den_matrix
        for i in range(len(config)):
            up, down = pattern[i + 1]

            if config[i] == "+":
                operator = up
                P_up, state_up = prob_den(operator, den_matrix_loop)
                # P_down, state_down = prob_den(down, den_matrix_loop)
                # norm = P_up + P_down
                # P_up = P_up / norm
                den_matrix_loop = state_up
                value = value + root / (2 ** (i + 1))
                prob_dict[str(value)] = P_up
            else:
                operator = down
                P_down, state_down = prob_den(operator, den_matrix_loop)
                # P_up, state_up = prob_den(up, den_matrix_loop)
                # norm = P_up + P_down
                # P_down = P_down / norm
                den_matrix_loop = state_down
                value = value - root / (2 ** (i + 1))
                prob_dict[str(value)] = P_down

    return prob_dict

def get_logical_op(prob_dict, S, M, numb_photons):
    observables = [1 for i in range(numb_photons + 1)]
    root = 2 ** (numb_photons)
    value = root
    # Following loop generates the measurement outcome for each qubit and appends it to "observables"
    for i in range(1, numb_photons + 1):
        value_left = value - root / (2 ** (i))
        value_right = value + root / (2 ** (i))
        if prob_dict[str(value_right)] > np.random.uniform():
            #print(prob_dict[str(value_right)])
            value = value_right
            observables[i] = observables[i] * 1

        else:
            value = value_left
            observables[i] = observables[i] * (-1)
    S_measured = []
    M_measured = 1
    # Two following loops generate the measured value of each stabilizer and logical X
    for stab in S:
        s = 1
        for i in range(len(stab)):
            if stab[i] != 'I':
                s = s * observables[i]
        S_measured.append(s)

    for i in range(len(M)):
        if M[i] != 'I':
            M_measured = M_measured * observables[i]



    return M_measured, S_measured



def old_corr(M_measured, S_measured):
    S3, S2, S1, S4, S5, S6, S7 = S_measured

    if S1 == -1 and S2 == -1 and S3 == -1 and S4 == 1 and S5 == 1 and S6 == 1 and M_measured == -1:
        M_measured = M_measured * (-1)
    elif S1 == 1 and S2 == 1 and S3 == 1 and S4 == 1 and S5 == 1 and S6 == 1 and M_measured == 1:
        M_measured = M_measured * (1)
    elif S1 == -1 and S2 == 1 and S3 == 1 and S4 == 1 and S5 == -1 and S6 == -1 and M_measured == 1:
        M_measured = M_measured * (1)
    elif S1 == 1 and S2 == -1 and S3 == 1 and S4 == -1 and S5 == 1 and S6 == -1 and M_measured == 1:
        M_measured = M_measured * (1)
    elif S1 == 1 and S2 == 1 and S3 == -1 and S4 == -1 and S5 == -1 and S6 == 1 and M_measured == 1:
        M_measured = M_measured * (1)
    elif S1 == 1 and S2 == -1 and S3 == -1 and S4 == 1 and S5 == -1 and S6 == -1 and M_measured == -1:
        M_measured = M_measured * (-1)
    elif S1 == -1 and S2 == 1 and S3 == -1 and S4 == -1 and S5 == 1 and S6 == -1 and M_measured == -1:
        M_measured = M_measured * (-1)
    elif S1 == -1 and S2 == -1 and S3 == 1 and S4 == -1 and S5 == -1 and S6 == 1 and M_measured == -1:
        M_measured = M_measured * (-1)

    return M_measured


def generate(S, M):
    """

    :param S: List of stabilizers
    :param M: Z, X or Y
    :return: meas. operators
    """
    B = get_B(S, M)
    pattern = meas_projectors(B)

    return pattern




def S_count_P_cond(prob_dict, S, M):
    S_measured = []
    S_count = {}
    P_dict = {}
    for i in range(monte_steps):  # Monte steps
        M_count, S_meas = get_logical_op(prob_dict, S, M, numb_photons=4)
        joined = ''.join(map(str, S_meas))
        if joined in S_measured:
            p_cond = conditional_M_S(M_count, S_meas)
            if p_cond == 1:
                P_dict[joined] = 1 + P_dict[joined]
            S_count[joined] = 1 + S_count[joined]
        else:
            S_measured.append(joined)
            p_cond = conditional_M_S(M_count, S_meas)
            if p_cond == 1:
                P_dict[joined] = 1
            else:
                P_dict[joined] = 0
            S_count[joined] = 1
    return P_dict, S_count



def heatmap(S_M_dict, monte_steps, kappas, T, keys, logical_op):
    for key in keys:
        M = S_M_dict[key][0]
        S = S_M_dict[key][1]
        print(M)
        print(S)
        ind_direct = logical_op
        pattern_X = generate(S, ind_direct)
        M_l = [[] for i in range(5)]
        C_l = [[] for i in range(5)]
        D_l = [[] for i in range(5)]
        configurations = get_configuration(5) # five for number 4 number of photons...
        for kappa in kappas:
            count1 = 0
            for T2 in T:
                den_matrix = get_five(int(key), kappa, T2)
                prob_dict = gen_prob_dict(pattern_X, den_matrix, configurations, numb_photons=4)
                counter = 0
                for i in range(10000):
                    M_count, S_meas = get_logical_op(prob_dict, S, M, numb_photons=4)
                    M_count = old_corr(M_count, S_meas)
                    if M_count == -1:
                        counter += 1 / 10000

                P_dict, S_count = S_count_P_cond(prob_dict, S, M)
                logical_err = logical_error(P_dict, S_count, monte_steps)
                logical_GHZ = GHZ_fid(kappa, logical_op, T2)
                norm = logical_GHZ + logical_err
                value = logical_GHZ - logical_err
                value = value / norm

                counter0 = (logical_GHZ - counter) / (logical_GHZ + counter)

                M_l[count1].append(value)
                C_l[count1].append(counter0)
                D_l[count1].append(value - counter0)
                count1 += 1
                print("P_dict {} and S_count {}".format(P_dict, S_count))
                print("logical error graph {}".format(logical_err))
                print("logical error graph of old decoder {}".format(counter))
                print("logical error GHZ {}".format(logical_GHZ))
                print("T2 {} and kappa {}".format(T2, kappa))

        p1 = sns.heatmap(M_l, yticklabels=["23.2", "33.2", "43.2", "53.2", "63.2"], \
                         xticklabels=["0.00105", "0.0021", "0.0105", "0.021", "0.105"], cmap="YlGnBu")
        plt.ylabel('$T_{2}^{*}$', fontsize=15)  # x-axis label with fontsize 15
        plt.xlabel('$\kappa$', fontsize=15)
        figure = p1.get_figure()
        figure.savefig('heatmap_graph_10_logical_Z.png', dpi=400)
        plt.show()
        p1 = sns.heatmap(C_l, yticklabels=["23.2", "33.2", "43.2", "53.2", "63.2"], \
                         xticklabels=["0.00105", "0.0021", "0.0105", "0.021", "0.105"], cmap="YlGnBu")
        plt.ylabel('$T_{2}^{*}$', fontsize=15)  # x-axis label with fontsize 15
        plt.xlabel('$\kappa$', fontsize=15)
        figure = p1.get_figure()
        figure.savefig('heatmap_graph_10_logical_Z_old_decoder.png', dpi=400)
        plt.show()

        p1 = sns.heatmap(D_l, yticklabels=["23.2", "33.2", "43.2", "53.2", "63.2"], \
                         xticklabels=["0.00105", "0.0021", "0.0105", "0.021", "0.105"], cmap="YlGnBu")
        plt.ylabel('$T_{2}^{*}$', fontsize=15)  # x-axis label with fontsize 15
        plt.xlabel('$\kappa$', fontsize=15)
        figure = p1.get_figure()
        figure.savefig('heatmap_graph_10_logical_Z_decoder_vs_decoder.png', dpi=400)
        plt.show()





def simulate(S_M_dict, monte_steps, kappas, T2, keys, logical_op):
    for key in keys:
        M = S_M_dict[key][0]
        S = S_M_dict[key][1]
        print(M)
        print(S)
        ind_direct = logical_op
        pattern_X = generate(S, ind_direct)
        M_l = []
        configurations = get_configuration(5) # five for number 4 number of photons...
        for kappa in kappas:
            den_matrix = get_five(int(key), kappa, T2)
            prob_dict = gen_prob_dict(pattern_X, den_matrix, configurations, numb_photons=4)
            P_dict, S_count = S_count_P_cond(prob_dict, S, M)
            logical_err = logical_error(P_dict, S_count, monte_steps)
            print("P_dict {} and S_count {}".format(P_dict, S_count))
            print(logical_err)
            M_l.append(logical_err)
        plt.plot(kappas, M_l, label=logical_op + "-" + key)
        plt.xscale('log')
        print(M_l[-1])

    M_GHZ = GHZ_fid(kappas, logical_op, T2)
    plt.plot(kappas, M_GHZ, label=logical_op + "-GHZ")
    plt.xlabel("$\kappa$")
    plt.ylabel("$\epsilon_{L}$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #path_codes = r"C:\Users\Admin\all_4_qubit_graphs.json"
    path_codes = r"C:\Users\Admin\now_with_lc_5_qubit.json"
    f = open(path_codes)
    the_dict = json.load(f)
    path_stabs = r"C:\Users\Admin\stab_and_logical_op_5_qubit.json"
    f = open(path_stabs)
    S_M_dict = json.load(f)
    # Few parameters
    kappas = [5*0.00021, 0.0021, 5*0.0021, 0.021, 0.021*5]
    monte_steps = 20000
    inputs = list(S_M_dict.keys())
    logical_op = "Z"
    inputs = ["10"]
    T2 = [23.2, 33.2, 43.2, 53.2, 63.2]
    # T2 = 23.2
    # Run the simulation given the parameters and the given 5-qubit graphs
    # simulate(S_M_dict, monte_steps, kappas, T2, inputs, logical_op)
    heatmap(S_M_dict, monte_steps, kappas, T2, inputs, logical_op)

    ## TODO:
    # Try not measuring the spin first, need more simulation steps
    # Try your old decoder for graph 10..
    # Write code for six qubit graphs
    # Change generatation of graphs to include stars..
    # Write out the transformations of star Z to star X(Y)
