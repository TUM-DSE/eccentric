from sim.circuit_4 import circuit
import multiprocessing as mp
import os
import itertools
from tqdm import tqdm
import numpy as np
import sys

# active = [0, 1, 2]
# active = 0 -> passive synchronization
# active = 1 -> active synchronization in the last round
# active = 2 -> No synchronization (ideal case)

shots = 100_000

def get_error(pd:tuple) -> float:
    active, p, ls_basis, d, idle = pd
    if active == 0:
        sync = 'passive'
    elif active == 1:
        sync = 'active'
    else:
        sync = None
    error = {3:0.001, 5:0.001, 7:0.001, 9:0.001, 11:0.001, 13:0.001, 15:0.001}
    sim = circuit(distance=d, num_patches_y=20, num_patches_x=20, spacing=1, disable_noise=False, fixed_t1=250, fixed_t2=150, fixed_cnot_latency=200, fixed_measure_latency=500, fixed_cnot_noise=error[d], fixed_measure_noise=error[d], rounds_per_op=d+1, idle_multiplier=3, basis=p, ls_basis=ls_basis, sync=sync, sync_idle=idle, sync_round=d+1, merge=True).from_string('qreg q[2];')
    
    e, _ = sim.get_error_rate(ckt=sim.ckt, num_shots=shots)
    return e

def get_error_google(pd:tuple) -> float:
    active, p, ls_basis, d, idle = pd
    if active == 0:
        sync = 'passive'
    elif active == 1:
        sync = 'active'
    else:
        sync = None
    error = {3:0.001, 5:0.001, 7:0.001, 9:0.001, 11:0.001, 13:0.001, 15:0.001}
    sim = circuit(distance=d, num_patches_y=20, num_patches_x=20, spacing=1, disable_noise=False, fixed_t1=25, fixed_t2=40, fixed_cnot_latency=50, fixed_measure_latency=660, fixed_cnot_noise=error[d], fixed_measure_noise=error[d], rounds_per_op=d+1, idle_multiplier=1, basis=p, ls_basis=ls_basis, sync=sync, sync_idle=idle, sync_round=d+1, merge=True).from_string('qreg q[2];')
    
    e, _ = sim.get_error_rate(ckt=sim.ckt, num_shots=shots)
    return e

if __name__ == '__main__':
    print(os.path.basename(__file__))
    if len(sys.argv) < 3:
        raise ValueError('Please provide the number of shots, and the number of processes')
    tshots = int(sys.argv[1])
    proc = int(sys.argv[2])
    nt = max(1, tshots // shots)
    
    err = {}
    p = ['Z']#['fenced', 'runahead']
    ls = ['Z']
    ds = range(3, 16, 2)
    active = [2]
    idles = [1000]
    pd = list(itertools.product(active, p, ls, ds, idles))
    
    es = np.zeros(len(pd))
    for i in tqdm(range(nt)):
        with mp.Pool(int(proc)) as pool:
            es += pool.map(get_error, pd)
            pool.close()
            pool.join()
    err = {i:j for i, j in zip(pd, es/nt)}
    err2 = {}
    # es = np.zeros(len(pd))
    # for i in tqdm(range(nt)):
    #     with mp.Pool(int(proc)) as pool:
    #         es += pool.map(get_error_google, pd)
    #         pool.close()
    #         pool.join()
    # err2 = {i:j for i, j in zip(pd, es/nt)}
    
    import pickle
    with open('active_passive_synchronization_ideal_zzxx.pkl', 'wb') as file:
        pickle.dump([err, err2], file)
