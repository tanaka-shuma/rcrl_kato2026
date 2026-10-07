# created by H. Kato, Oita Univ., Japan
# <last modified 13:40 14-Jul-2022 JST>

import numpy as np
import argparse

def get_opts():
    parms={'T': 1000, 'dt': 0.2, 'N': np.array([800, 0, 200, 0]), 'mod': np.array([1,1]),
           'g': np.array([[[[5e-6, 5e-6], [2e-5, 1.5e-5]], [[.0, .0], [.0, .0]]], [[[.0, .0], [.0, .0]], [[.0, .0], [.0, .0]]]]),
           'D': np.array([[[1.0, 4.0],[1.0, 1.2]], [[2.0, 3.0],[0.8, 0.8]]]), 'Vrev': np.array([.0, -75.0]),
           'Tsyn': np.array([2.0, 5.0]), 'Tsyn_slow': np.array([150.0, 200.0]),
           'g0': np.array([[2.000000e-04, 0, 1.270000e-04, 0],[1.163636e-03, 0, 5.080000e-04, 0]]), 'sigma': np.array([[8.000000e-05, 0, 4.343400e-05, 0],[1.861818e-04, 0, 6.949440e-05, 0]]),  # 0.01 spk/s if no connections (new parameter search)
           'U': np.array([[0.21,0.3],[0.14, 0.3]]), 'tau_f': np.array([[5.0, 5.0],[2.0, 2.0]]), 'tau_d': np.array([[463.0, 227.0],[875.0, 400.0]]),
           'gk': 1.70e-5, 'Cstep': 0.1, 'tca': 2550,   # E-neuron Ca-dependent K adaptation: depth (gk/Cstep) and Ca time constant (tca, ms). Default = the single size-independent (gk,tca) that puts burst dur onto the N-increasing ideal band across ALL sizes at the confirmed flat-rate coupling, keeping rate ~flat (CV 5%) with I co-bursting (gktca refine scan 2026-07-19).
           'mu_inter': 2.0, 'sig_inter': 0.0,            
           'spk': False, 'idx_flag': False,
           'seed': 1}

    parser = argparse.ArgumentParser(description='parameters for a spiking neural network')
    
    parser.add_argument('-seed', '--seed', help='reset seed of random numbers', type=int)
    parser.add_argument('-T', '--time', help='simulation time', type=int)
    parser.add_argument('-dt', '--time_res', help='time resolution of simulation', type=float)
    parser.add_argument('-Ne', '--num_e', help='# of excitatory neurons in a network', type=int)
    parser.add_argument('-Ni', '--num_i', help='# of excitatory neurons in a network', type=int)
    parser.add_argument('-r', '--Mrow', help='# of modules in row on a lattice', type=int)
    parser.add_argument('-c', '--Mcol', help='# of modules in column on a lattice', type=int)
    parser.add_argument('-Minter', '--mu_inter', help='# of axones from a module to another', type=float)
    parser.add_argument('-Sinter', '--sig_inter', help='SD of connections between modules', type=float)    
    parser.add_argument('-g_ee', '--g_ee_intra', help='synaptic strength between E and E neurons within each module', type=float)
    parser.add_argument('-g_ei', '--g_ei_intra', help='synaptic strength between E and I neurons within each module', type=float)
    parser.add_argument('-g_ie', '--g_ie_intra', help='synaptic strength between I and E neurons within each module', type=float)
    parser.add_argument('-g_ii', '--g_ii_intra', help='synaptic strength between I and I neurons within each module', type=float)
    parser.add_argument('-g_inter', '--g_ee_inter', help='synaptic strength between modules (between E neurons only)', type=float)
    parser.add_argument('-g_ee_slow', '--g_ee_slow_intra', help='synaptic strength for slow currents between E and E neurons within each module', type=float)
    parser.add_argument('-g_ei_slow', '--g_ei_slow_intra', help='synaptic strength for slow currents between E and I neurons within each module', type=float)
    parser.add_argument('-g_ie_slow', '--g_ie_slow_intra', help='synaptic strength for slow currents between I and E neurons within each module', type=float)
    parser.add_argument('-g_ii_slow', '--g_ii_slow_intra', help='synaptic strength for slow currents between I and I neurons within each module', type=float)
    parser.add_argument('-g_slow_inter', '--g_ee_slow_inter', help='synaptic strength for slow currents between modules (between E neurons only)', type=float)
    parser.add_argument('-spk', '--spike_data', help='flag for output spike data', action='store_true')
    parser.add_argument('-idx', '--neuron_index', help='focused neuron index', type=int)

    args = parser.parse_args()

    if args.seed!=None:
        parms['seed'] = args.seed
    if args.time!=None:
        parms['T'] = args.time
    if args.time_res!=None:
        parms['dt'] = args.time_res
    if args.num_e!=None:
        parms['N'][0] = args.num_e
    if args.num_i!=None:
        parms['N'][2] = args.num_i
    if args.Mrow!=None:
        parms['mod'][1] = args.Mrow
    if args.Mcol!=None:
        parms['mod'][0] = args.Mcol
    if args.g_ee_intra!=None:
        parms['g'][0,0,0,0] = args.g_ee_intra
    if args.g_ei_intra!=None:
        parms['g'][0,0,0,1] = args.g_ei_intra
    if args.g_ie_intra!=None:
        parms['g'][0,0,1,0] = args.g_ie_intra
    if args.g_ii_intra!=None:
        parms['g'][0,0,1,1] = args.g_ii_intra
    if args.g_ee_inter!=None:
        parms['g'][0,1,0,0] = args.g_ee_inter
    if args.g_ee_slow_intra!=None:
        parms['g'][1,0,0,0] = args.g_ee_slow_intra
    if args.g_ei_slow_intra!=None:
        parms['g'][1,0,0,1] = args.g_ei_slow_intra
    if args.g_ie_slow_intra!=None:
        parms['g'][1,0,1,0] = args.g_ie_slow_intra
    if args.g_ii_slow_intra!=None:
        parms['g'][1,0,1,1] = args.g_ii_slow_intra
    if args.g_ee_slow_inter!=None:
        parms['g'][1,1,0,0] = args.g_ee_slow_inter
    if args.mu_inter!=None:
        parms['mu_inter'] = args.mu_inter
    if args.sig_inter!=None:
        parms['sig_inter'] = args.sig_inter        
    if args.spike_data!=None:
        parms['spk'] = args.spike_data
    if args.neuron_index!=None:
        parms['idx_flag'] = True
        parms['idx'] = args.neuron_index

    return parms

        
# EOF
