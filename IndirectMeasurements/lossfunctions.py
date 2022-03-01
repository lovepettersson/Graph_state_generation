import numpy as np
import qecc as q
from LossTolerance.ErrorCorrectionFunctions.EC_DecoderClasses import *
from LossTolerance.CodesFunctions.graphs import *


def conditional_M_S(M):
    if M == -1:  # Currently only concerend with up in respective bases
        return 1
    else:
        return 0


def lost_qubits(observables, loss):
    lost_qubits = []
    new_observables = []
    for i in range(len(observables)):
        if i == 0:
            continue
        else:
            if loss > np.random.uniform():
                lost_qubits.append(i)
    for i in range(len(observables)):
        if i in lost_qubits:
            continue
        else:
            new_observables.append(observables[i])

    return new_observables, lost_qubits

def new_graph(graph, observables):
    if graph == "star-middle":
        graph = gen_star_graph(len(observables))
        gstate = GraphState(graph)
    elif graph == "fully":
        graph = gen_fullyconnected_graph(len(observables))
        gstate = GraphState(graph)
    else:
        graph = gen_star_graph(len(observables), 1)
        gstate = GraphState(graph)
    return gstate


def get_stabs_logical_op(gstate, ind_meas_op, in_qubit=0):
    all_stabs = [this_op.op for this_op in q.from_generators(gstate.stab_gens)]
    all_stabs.remove('I' * len(all_stabs[0]))
    filtered_stabs_ind_meas = filter_stabs_input_op_compatible(all_stabs, ind_meas_op, in_qubit)
    # print(filtered_stabs_ind_meas)
    ### identify all possible indirect measurement families
    meas_families = find_ind_meas_EC_families(filtered_stabs_ind_meas, ind_meas_op, in_qubit)
    best_meas = max(meas_families, key=lambda x: len(meas_families[x][1]))
    best_fam = meas_families[best_meas]
    logical_op = [best_fam[0]]
    stabs = best_fam[1]
    return logical_op, stabs


def get_logical_op_loss(prob_dict, numb_photons, loss, graph, ind_meas_op):
    observables = [1 for i in range(numb_photons + 1)]
    root = 2 ** (numb_photons)
    value = root
    # Following loop generates the measurement outcome for each qubit and appends it to "observables"
    for i in range(1, numb_photons + 1):
        value_left = value - root / (2 ** (i))
        value_right = value + root / (2 ** (i))
        if prob_dict[str(value_right)] > np.random.uniform():
            # print(prob_dict[str(value_right)])
            value = value_right
            observables[i] = observables[i] * 1

        else:
            value = value_left
            observables[i] = observables[i] * (-1)

    observables, l_qubits = lost_qubits(observables, loss)
    if len(observables) == 1:
        return 0, [0]
    else:
        if graph == "spin-leaf":
            if 1 in l_qubits:
                return 1/2, 1/2
            else:
                gstate = new_graph(graph, observables)
                M, S = get_stabs_logical_op(gstate, ind_meas_op)
                M = M[0]
                S_measured = []
                M_measured = 1
                # Two following loops generate the measured value of each stabilizer and logical M
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
        else:
            gstate = new_graph(graph, observables)
            M, S = get_stabs_logical_op(gstate, ind_meas_op)
            M = M[0]
            S_measured = []
            M_measured = 1
            # Two following loops generate the measured value of each stabilizer and logical M
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


def S_count_P_cond_loss(prob_dict, numb_photons, loss, graph, ind_meas_op, monte_steps):
    S_measured = []
    S_count = {}
    P_dict = {}
    steps = 0
    for i in range(monte_steps):  # Monte steps
        M_count, S_meas = get_logical_op_loss(prob_dict, numb_photons, loss, graph, ind_meas_op)
        if M_count == 0:
            steps += 0
        else:
            steps += 1
            joined = ''.join(map(str, S_meas))
            if joined in S_measured:
                p_cond = conditional_M_S(M_count)
                if p_cond == 1:
                    P_dict[joined] = 1 + P_dict[joined]
                S_count[joined] = 1 + S_count[joined]
            else:
                S_measured.append(joined)
                p_cond = conditional_M_S(M_count)
                if p_cond == 1:
                    P_dict[joined] = 1
                else:
                    P_dict[joined] = 0
                S_count[joined] = 1
    return P_dict, S_count, steps