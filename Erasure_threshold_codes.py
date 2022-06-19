import matplotlib.pyplot as plt
from CodesFunctions.graphs import *
from LossToleranceFunctions.LT_Decoders_Classes import LT_FullDecoder, LT_IndMeasDecoder
import json
import itertools
from LossToleranceFunctions.Losstolerant_fusion import*

import numpy as np
import qecc as q
import networkx as nx


class ErasureDecoder(object):

    def __init__(self, gstate, indmeas_pauli, p_fail, transmission, w, in_qubit=0):
        # Initialize the decoder
        self.gstate = gstate
        self.indmeas_pauli = indmeas_pauli
        self.in_qubit = in_qubit
        self.p_fail = p_fail
        self.transmission = transmission
        self.w = w
        self.all_stabs = self.remove_paulis()
        self.configurations = self.get_configuration()
        self.succ_config = self.get_succ_conf()
        self.pattern = self.gen_meas_pattern()
        self.p_x = self.er_x()
        self.p_z = self.er_z()


    def get_indirect_meas_stabs(self):
        num_qubits = len(self.gstate)
        # print("num_qubits", num_qubits)
        # get all possible 2^N stabilizers.
        # TODO: to be improved checking only "smart" stabilizers.
        all_stabs = q.from_generators(self.gstate.stab_gens)
        poss_stabs_list = []
        for this_stab0 in all_stabs:
            this_stab = this_stab0.op
            if this_stab[self.in_qubit] == self.indmeas_pauli:
                meas_weight = num_qubits - this_stab.count('I')
                Z_weight = this_stab.count('Z') / (meas_weight + 1)
                meas_qubits = [qbt for qbt in range(num_qubits)
                               if ((qbt != self.in_qubit) and (this_stab[qbt] != 'I'))]
                ### order them such that we always prefer mstrategies with smaller weight, and with more Zs in the non-trivial paulis.
                poss_stabs_list.append([this_stab, meas_qubits, meas_weight, Z_weight])
        poss_stabs_list.sort(key=lambda x: x[2] - x[3])
        return poss_stabs_list

    def remove_paulis(self):
        new_stabs = []
        for stab in self.get_indirect_meas_stabs():
            Y_count = 0
            for idx in stab[0]:
                if idx == "Y":
                    Y_count += 1
            if Y_count == 0:
                new_stabs.append(stab)
        return new_stabs

    def get_configuration(self):
        numb = len(self.all_stabs[0][0])-1
        configurations = []
        iter = product('+-', repeat=numb)
        for ch in iter:
            configurations.append(ch)
        return configurations

    def get_succ_conf(self):
        config = []
        for stab in self.all_stabs:
            config.append(stab[1])
        return config

    def gen_meas_pattern(self):
        pattern = {}
        for stabs in self.all_stabs:
            for qbits in stabs[1]:
                if qbits in pattern.keys():
                    continue
                else:
                    pattern[qbits] = (stabs[0][qbits])
        return pattern

    def er_z(self):
        return (1 - (1 - self.w * self.p_fail) * (self.transmission ** (1 / self.p_fail)))

    def er_x(self):
        return (1 - (1 - (1 - self.w) * self.p_fail) * (self.transmission ** (1 / self.p_fail)))

    def run_dec(self):
        tot_term = 0
        for config in self.configurations:
            succ_meas = []
            failed_meas = []
            for i in range(len(config)):
                if config[i] == "+":
                    succ_meas.append(i + 1)
                else:
                    failed_meas.append(i + 1)
            for set_in_list in self.succ_config:
                set_in_list = set(set_in_list)
                succ_meas = set(succ_meas)
                if set_in_list.issubset(succ_meas):
                    term = 1
                    for qbit in succ_meas:
                        if self.pattern[qbit] == "X":
                            term = term * (1 - self.p_x)
                        else:
                            term = term * (1 - self.p_z)

                    for qbit in failed_meas:

                        if self.pattern[qbit] == "X":
                            term = term * (self.p_x)
                        else:
                            term = term * (self.p_z)
                    tot_term += term
                    break
        return tot_term
'''
def erasure_z(p_fail, t, w):
    return (1 - (1-w*p_fail)*(t**(1/p_fail)))


def erasure_x(p_fail, t, w):
    return (1 - (1-(1-w)*p_fail)*(t**(1/p_fail)))

def gen_erase_prob_variable(p_z, p_x, configurations, succ_conf, pattern):
    tot_term = 0
    for config in configurations:
        succ_meas = []
        failed_meas = []
        for i in range(len(config)):
            if config[i] == "+":
                succ_meas.append(i+1)
            else:
                failed_meas.append(i+1)
        for set_in_list in succ_conf:
            set_in_list = set(set_in_list)
            succ_meas = set(succ_meas)
            if set_in_list.issubset(succ_meas):
                term = 1
                for qbit in succ_meas:
                    if pattern[qbit] == "X":
                        term = term * (1-p_x)
                    else:
                        term = term* (1- p_z)
                for qbit in failed_meas:
                    if pattern[qbit] == "X":
                        term = term * (p_x)
                    else:
                        term = term* (p_z)
                tot_term += term
                break
    return tot_term


def run_new_erase_decoder_zx(gstate, p_fail, loss, s):
    values = []
    decod0 = LT_IndMeasDecoder(gstate, 'X', in_qubit)
    stab_X = decod0.all_stabs
    stab_X = remove_Y(stab_X)
    decod1 = LT_IndMeasDecoder(gstate, 'Z', in_qubit)
    stab_Z = decod1.all_stabs
    stab_Z = remove_Y(stab_Z)
    configurations = get_configuration(len(stab_X[0][0])-1)
    succ_config_Z = get_succ_conf(stab_Z)
    succ_config_X = get_succ_conf(stab_X)
    pattern_Z = gen_meas_pattern(stab_Z)
    pattern_X  = gen_meas_pattern(stab_X)
    for t in loss:
        p_x = erasure_x(p_fail, t, s)
        p_z = erasure_z(p_fail, t, s)
        succ_X = gen_erase_prob_variable(p_z, p_x, configurations, succ_config_X, pattern_X)
        succ_Z = gen_erase_prob_variable(p_z, p_x, configurations, succ_config_Z, pattern_Z)
        fail_X = 1 - succ_X
        fail_Z = 1 - succ_Z
        print(succ_X)
        average = (fail_X + fail_Z) / 2
        values.append(average)
    return values
'''
if __name__ == '__main__':

    graph = gen_random_connected_graph(6)
    gstate = GraphState(graph)
    p_fail = 1 / 2
    transmission = 1
    weigth = np.linspace(0.001, 1, 20)
    transmissions = np.linspace(0.8, 1, 10)
    average_best = 0
    weigths = []
    best_erasures = []
    loss_value = []
    for transmission in transmissions:
        for wt in weigth:
            decoder_Z = ErasureDecoder(gstate, "Z", p_fail, transmission, wt)
            decoder_X = ErasureDecoder(gstate, "X", p_fail, transmission, wt)
            succ_Z = decoder_Z.run_dec()
            succ_X = decoder_X.run_dec()
            average = (succ_X + succ_Z) / 2
            if average > average_best:
                average_best = average
                print("average {}".format(average))
                print("weigth {}".format(wt))
                weigths.append(wt)
                best_erasures.append(average)
                loss_value.append(transmission)
    plt.plot(loss_value, weigths, label="weigths")
    plt.plot(loss_value, best_erasures, label="erasure_prob")
    plt.xlabel("Transmission")
    plt.legend()
    plt.show()


    '''
    in_qubit = 0
    ring_network_threshold = 0.1198
    p_fail = 1 / 4
    transmission = [0.98]
    weigth = 0.001
    arms = [3, 4, 5, 6, 7, 8, 9, 10]
    actual_arms = [i - 1 for i in arms]
    eras_plot_vals = []
    thresold_ring = []
    for i in arms:
        graph = gen_star_graph(i, 0)
        gstate = GraphState(graph)
        decoder = ErasureDecoder(gstate, "X", 1/4, 0.98, 0.001)
        print("my decoder {}".format(decoder.run_dec()))
        encoded_erasure = run_new_erase_decoder_zx(gstate, p_fail, transmission, weigth)[0]
        thresold_ring.append(ring_network_threshold)
        eras_plot_vals.append(encoded_erasure)
        print(encoded_erasure)
    plt.plot(actual_arms, eras_plot_vals, "-o", label="encoded-GHZ")
    plt.plot(actual_arms, thresold_ring, "k:", label="ring-network-threshold-value")
    plt.title("$p_{fail} =$" + " {}, t = {} % and weigth = {}".format(p_fail, transmission[0], weigth))
    plt.xlabel("Number of arms")
    plt.ylabel("Encoded erasure prob.")
    plt.legend()
    plt.show()
    '''
