from sim.circuit_4 import circuit
import multiprocessing as mp
import os
import itertools
from tqdm import tqdm
import numpy as np
import pickle
import sys

shots = 131_072

def get_error(pd:tuple) -> float:
    active, p, ls_basis, d, idle, rounds, orig_slack, c, eps = pd
    if active == 0:
        sync = 'passive'
    elif active == 1:
        sync = 'active'
    else:
        sync = 'active'
        idle = 0
    if rounds is None:
        rounds = d + 1
    error = {3:0.001, 5:0.001, 7:0.001, 9:0.001, 11:0.001, 13:0.001, 15:0.001}
    sim = circuit(distance=d, num_patches_y=20, num_patches_x=20, spacing=1, disable_noise=False, fixed_t1=200, fixed_t2=150, fixed_cnot_latency=50, fixed_measure_latency=500, fixed_cnot_noise=error[d], fixed_measure_noise=error[d], rounds_per_op=d + 1, pre_ls_rounds=d + rounds + 1, post_ls_rounds=d + 1, idle_multiplier=3, basis=p, ls_basis=ls_basis, sync=sync, sync_idle=idle, sync_round=d + 1, merge=True).from_string('qreg q[2];')
    
    e, _ = sim.get_error_rate(ckt=sim.ckt, num_shots=shots)
    return e

def get_error_er(pd:tuple) -> float:
    active, p, ls_basis, d, idle, rounds, orig_slack, c, eps = pd
    if active == 0:
        sync = 'passive'
    elif active == 1:
        sync = 'active'
    else:
        sync = 'passive'
    if rounds is None:
        rounds = d + 1
    error = {3:0.001, 5:0.001, 7:0.001, 9:0.001, 11:0.001, 13:0.001, 15:0.001}
    sim = circuit(distance=d, num_patches_y=20, num_patches_x=20, spacing=1, disable_noise=False, fixed_t1=200, fixed_t2=150, fixed_cnot_latency=50, fixed_measure_latency=500, fixed_cnot_noise=error[d], fixed_measure_noise=error[d], rounds_per_op=d + 1, pre_ls_rounds=d + rounds + 1, post_ls_rounds=d + 1, idle_multiplier=3, basis=p, ls_basis=ls_basis, sync=sync, sync_idle=0, sync_round=d + 1, merge=True).from_string('qreg q[2];')
    
    e, _ = sim.get_error_rate(ckt=sim.ckt, num_shots=shots)
    return e

def get_error_quera(pd:tuple) -> float:
    active, p, ls_basis, d, idle, rounds, orig_slack, c, eps = pd
    if active == 0:
        sync = 'passive'
    elif active == 1:
        sync = 'active'
    else:
        sync = 'active'
        idle = 0
    if rounds is None:
        rounds = d + 1
    error = {d:0.003 for d in range(3, 16, 2)}
    sim = circuit(distance=d, num_patches_y=20, num_patches_x=20, spacing=1, disable_noise=False, fixed_t1=4 * 10**6, fixed_t2=1.5*10**6, fixed_cnot_latency=200*10**3, fixed_measure_latency=1 * 10**6, fixed_cnot_noise=error[d], fixed_measure_noise=0.002, rounds_per_op=d + 1, pre_ls_rounds=d + rounds + 1, post_ls_rounds=d + 1, idle_multiplier=1, basis=p, ls_basis=ls_basis, sync=sync, sync_idle=idle, sync_round=d + rounds + 1, merge=True).from_string('qreg q[2];')
    
    e, _ = sim.get_error_rate(ckt=sim.ckt, num_shots=shots)
    return e

def find_extra_rounds(c1, c2, slack):
    flag = True
    m = 0
    while flag:
        m += 1
        if (m * c1 + slack) % c2 == 0:
            flag = False
            return m
        if m > 100:
            return -1

def find_hybrid_param(c1, c2, slack, eps):
    flag = True
    z = 0
    while flag:
        z += 1
        t1 = c1 * z + slack
        r = np.ceil(t1 / c2) * c2 - t1
        if (t1 + r) % c2 == 0 and r < eps:
            flag = False
            return z, r
        if z > 50:
            return -1, -1

if __name__ == "__main__":
    print(os.path.basename(__file__))
    if len(sys.argv) < 3:
        raise ValueError('Please provide the number of shots, and the number of processes')
    tshots = int(sys.argv[1])
    proc = int(sys.argv[2])
    nt = max(1, tshots // shots)
    
    offsets = [0.2, 0.4, 0.6]
    slacks = [0.2, 0.6, 1, 1.6, 2]
    c1 = 2 * 10 ** 6 # 2 ms - 2M ns
    offsets = [i * 10 **6 for i in offsets]
    slacks = [i * 10 ** 6 for i in slacks]
    combs = list(itertools.product(offsets, slacks))
    epsilons = [0.1, 0.4]
    epsilons = [i * 10**6 for i in epsilons]
    ds = [11]
    active = [1]
    
    
    # For every comb, find the LER of active, extra rounds, hybrid policies 
    inputs_er = [[c1, c1 + offset, slack, find_extra_rounds(c1, c1+offset, slack)] 
                 for offset, slack in combs if find_extra_rounds(c1, c1+offset, slack) != -1]
    combs = itertools.product(offsets, slacks, epsilons)
    inputs_hybrid = []
    for offset, slack, eps in combs:
        z, r = find_hybrid_param(c1, c1 + offset, slack, eps)
        if z == -1:
            continue
        inputs_hybrid.append([c1, c1 + offset, slack, eps, z, r])
        pass
    
    # Compute LERs
    # Create all test inputs
    inputs = []
    for basis in ['X', 'Z']:
        for d in ds:
            for active in [0, 1, 2, 3]:
                if active == 2:
                    for inp in inputs_er:
                        inputs.append(tuple((active, basis, basis, d, 0, inp[3], inp[2], inp[1], 0)))
                    pass
                if active in [1, 0]:
                    for inp in inputs_er:
                        inputs.append(tuple((active, basis, basis, d, inp[2], 0, inp[2], inp[1], 0)))
                    pass
                if active == 3:
                    for inp in inputs_hybrid:
                        inputs.append(tuple((active, basis, basis, d, inp[-1], inp[-2], inp[-4], inp[1], inp[-3])))
                    pass
                pass
        pass
    print(len(inputs))
    
    # # Print inputs to file for batch runs
    # for input in inputs:
    #     s = ','.join(['%s'%(i) for i in input])
    #     file = open('hybrid_batch_inputs.txt', '+a')
    #     file.write(s+'\n')
    #     file.close()
    
    es = np.zeros(len(inputs))
    for i in tqdm(range(nt)):
        with mp.Pool(int(proc)) as pool:
            es += pool.map(get_error_quera, inputs)
            pool.close()
            pool.join()
    err = {i:j for i, j in zip(inputs, es/nt)}
    with open('neutral_atom_hybrid.pkl', 'wb') as file:
        pickle.dump([err, inputs_er, inputs_hybrid], file)