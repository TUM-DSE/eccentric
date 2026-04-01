import numpy as np
import networkx as nx
import itertools
from circuit_parser import circuit_parser
from gate_lib import gate_lib
from stat_counter import stat_counter
from stim_integ import stim_integ

# Work at an abstraction level of patches -- the input to this class is a circuit
# defining the logical qubits and the logical operations between them.
# Will need a way to parse the input circuit (from a file or input string).
# Once all logical qubits have been defined, logical operations using lattce surgery
# can be performed. 

# Some constraints to make implementation simpler:
# - Z boundaries fixed to be on the left and right side of the patches
# - X boundaries fixed to be on the top and bottom of the patches
# So a merge/split of two patches in the vertical direction will operate on the X boundary
# and ones in the horizontal direction will operate on the Z boundary.
class circuit(circuit_parser, gate_lib, stat_counter, stim_integ):
    def __init__(self, distance:int=3, disable_noise:bool=False, seed:int=0,
                 num_patches_x:int=20, num_patches_y:int=20, 
                 spacing:int=1, rounds_per_op:int=None,
                 init_error=None,
                 fixed_measure_noise:float=None, 
                 fixed_measure_latency:float=None,
                 fixed_cnot_noise:float=None,
                 fixed_cnot_latency:float=None,
                 fixed_t1:float=None, fixed_t2:float=None,
                 cnot_latency_dist:str=None,
                 cnot_error_dist:str=None,
                 measure_latency_dist:str=None,
                 measure_error_dist:str=None,
                 cnot_latency_mean:float=None,
                 cnot_latency_std:float=None,
                 cnot_error_mean:float=None,
                 cnot_error_std:float=None,
                 measure_latency_mean:float=None,
                 measure_latency_std:float=None,
                 measure_error_mean:float=None,
                 measure_error_std:float=None,
                 idle_multiplier:int=1,
                 error1Q:float=0.0001, latency1Q:float=30,
                 gates_1Q:list=['H', 'X', 'S', 'Z', 'Y'],
                 gates_2Q:list=['CX', 'CZ'],
                 measures:list=['M', 'MR', 'MX']) -> None:
        np.random.seed(seed)
        self.d = distance
        self.disable_noise = disable_noise
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        assert spacing > 0, 'expected spacing between patches > 0, got %i'%(spacing)
        self.spacing = spacing
        self.init_error = init_error
        self.coords_index_mapper = {}
        self.index_coords_mapper = {}
        self.measurement_tracker = []
        self.pre_decode = False
        self.distributed_idle = False
        self.distributed_rounds = 1
        self.total_idle = 0
        self.fixed_t1 = fixed_t1
        self.fixed_t2 = fixed_t2
        self.gates = gates_1Q + gates_2Q + measures
        self.time = 0 # Tracks the time of the fastest qubit after every TICK
        self.decode_latency = 100 # In ns
        if rounds_per_op == None:
            rounds_per_op = distance
        self.rounds_per_op = rounds_per_op
        circuit_parser.__init__(self)
        gate_lib.__init__(self, distance, num_patches_x, num_patches_y,
                          spacing, seed=seed, fixed_cnot_latency=fixed_cnot_latency,
                          fixed_measure_noise=fixed_measure_noise,
                          fixed_cnot_noise=fixed_cnot_noise,
                          fixed_measure_latency=fixed_measure_latency,
                          cnot_latency_dist=cnot_latency_dist,
                          cnot_error_dist=cnot_error_dist,
                          measure_latency_dist=measure_latency_dist,
                          cnot_latency_mean=cnot_latency_mean,
                          cnot_latency_std=cnot_latency_std,
                          cnot_error_mean=cnot_error_mean,
                          cnot_error_std=cnot_error_std,
                          measure_error_mean=measure_error_mean,
                          measure_error_std=measure_error_std,
                          measure_latency_mean=measure_latency_mean,
                          measure_latency_std=measure_latency_std,
                          measure_error_dist=measure_error_dist,
                          idle_multiplier=idle_multiplier,
                          error1Q=error1Q, latency1Q=latency1Q,
                          gates_1Q=gates_1Q, gates_2Q=gates_2Q, measures=measures)
        pass
    
    def get_patch_layout(self) -> None:
        # Print the layout of all patches for the given code distance and grid
        # size.
        # Every logical qubit spans 2*d points on the grid
        # The spacing defines the number of data qubits between every patch
        for i in range(self.num_patches_x):
            for j in range(self.num_patches_y):
                space = '.' * self.spacing
                if j == self.num_patches_y - 1:
                    space = ''
                print('|%i|'%((i) * (self.num_patches_y) + (j + 1)) + space, end='')
                pass
            print('')
        return
    
    def noise_profile(self) -> None:
        # Define the noise affecting every qubit in the lattice
        # This will be added during the circuit synthesis phase.
        return
    
    def map_qubit(self, qubit:int=0) -> dict:
        # Place data qubits
        data_coords = []
        # Place measurement qubits
        x_measure_coords = []
        z_measure_coords = []
        # Need to define which data qubits form a patch
        # Find row and column of the patch specified by qubit
        row = np.floor((qubit) / self.num_patches_y)
        col = (qubit) % self.num_patches_y
        patch_start_x = col * (2 * self.d + self.spacing * 2)
        patch_start_y = row * (2 * self.d + self.spacing * 2)
        # Shift origin temporarily to treat this patch as a single patch
        mapper = {}
        for i in range(self.d * 2 + 1):
            for j in range(self.d * 2 + 1):
                if i % 2 == j % 2:
                    indices = ((int(patch_start_x + i), int(patch_start_y + j)))
                    mapper[tuple((i, j))] = indices
                pass
        # Define measure qubits for both basis
        for i in range(self.d + 1):
            for j in range(self.d + 1):
                coords = tuple((2 * i, 2 * j))
                left_right_boundary = (i == 0 or i == self.d)
                top_bottom_boundary = (j == 0 or j == self.d)
                parity = i % 2 != j % 2
                if left_right_boundary and parity:
                    continue
                if top_bottom_boundary and not parity:
                    continue
                if parity:
                    x_measure_coords.append(coords)
                else:
                    z_measure_coords.append(coords)
                pass
        # Define observables and data qubits
        x_observable = []
        z_observable = []
        for i in range(self.d * 2 + 1): # vertical
            for j in range(self.d * 2 + 1): # horizontal
                if i % 2 == 1 and j % 2 == 1:
                    data_coords.append(tuple((i, j)))
                    if i == 1:
                        x_observable.append(tuple((i, j)))
                    if j == 1:
                        z_observable.append(tuple((i, j)))
        # Shift origin back
        data_coords = [mapper[i] for i in data_coords]
        x_observable = [mapper[i] for i in x_observable]
        z_observable = [mapper[i] for i in z_observable]
        x_measure_coords = [mapper[i] for i in x_measure_coords]
        z_measure_coords = [mapper[i] for i in z_measure_coords]

        all_coords = data_coords + x_measure_coords + z_measure_coords

        # Assign T1 and T2 times to each coord (physical qubit)
        _ = [self.assign_T1(coord, val=self.fixed_t1) for coord in all_coords]
        _ = [self.assign_T2(coord, val=self.fixed_t2) for coord in all_coords]

        # Pass coords of this logical qubit to the final circuit synthesizer
        mappings = {}
        mappings['data_coords'] = data_coords
        mappings['x_observable'] = x_observable
        mappings['z_observable'] = z_observable
        mappings['x_measure_coords'] = x_measure_coords
        mappings['z_measure_coords'] = z_measure_coords
        return mappings
    
    def coords_to_index_mapper(self, coords:dict) -> None:
        coords_to_index = lambda coord: int(((coord[0] + coord[1]) * 
                                             (coord[0] + coord[1] + 1)) // 
                                             2 + coord[1]) # Cantor pairing
        coords_index_mapper = {}
        index_coords_mapper = {}
        measure_idxs = []
        data_idxs = []
        self.qubit_patch_mapper = {}
        self.patch_qubit_mapper = {}
        patch = 0
        if type(coords) != dict:
            raise ValueError('Expected a dictionary of coordinate dictionaries per logical qubit')
        # Proceed through every stage of the surface code cycle for every qubit all at once. 
        for mappings in coords.values():
            if type(mappings) != dict:
                raise ValueError('Expected a dictionary of coordinate dictionaries per logical qubit')
            data_coords = mappings['data_coords']
            x_measure_coords = mappings['x_measure_coords']
            z_measure_coords = mappings['z_measure_coords']
            all_coords = data_coords + x_measure_coords + z_measure_coords
            measure_idxs += [coords_to_index(c) for c in x_measure_coords + z_measure_coords]
            data_idxs += [coords_to_index(c) for c in data_coords]
            for c in all_coords:
                idx = coords_to_index(c)
                coords_index_mapper[c] = idx
                index_coords_mapper[idx] = c
                self.qubit_patch_mapper[idx] = patch
                self.patch_qubit_mapper[patch] = [idx] if patch not in self.patch_qubit_mapper.keys() \
                    else self.patch_qubit_mapper[patch] + [idx] 
            patch += 1
        self.coords_index_mapper = coords_index_mapper # Save for the future
        self.index_coords_mapper = index_coords_mapper
        self.measure_idxs = measure_idxs
        self.data_idxs = data_idxs
        self.add_skew()
        return

    def get_qubit_idxs(self) -> list:
        return list(self.index_coords_mapper.keys())
    
    def get_qubit_coords(self) -> list:
        return list(self.coords_index_mapper.keys())
    
    def get_measure_qubit_coords(self) -> list:
        return [self.index_coords_mapper[idx] for idx in self.measure_idxs]
    
    def get_patch_idxs(self) -> list:
        return list(self.patch_qubit_mapper.keys())
    
    # A tick represents the start of a new frame. 
    def tick(self, str: str) -> str:
        t = str + 'TICK\n'
        return t 
    
    def __X():
        return 'X'
    
    def __Z():
        return 'Z'
    
    def reset_data(self, mappings:dict) -> str:
        ckt_str = ''
        data_coords = mappings['data_coords']
        data_idx = [self.coords_index_mapper[i] for i in data_coords]
        if True: # replace with condition for X or Z basis experiment
            reset_data = 'R'
        else:
            reset_data = 'RX'
        ckt_str += reset_data + ' ' + ' '.join(str(i) for i in data_idx) + "\n"
        if self.init_error != None:
            if True:
                ckt_str += 'X_ERROR(%f) '%(self.init_error) + ' '.join(str(i) for i in data_idx) + "\n"
            else:
                ckt_str += 'Z_ERROR(%f) '%(self.init_error) + ' '.join(str(i) for i in data_idx) + "\n"
            pass
        return ckt_str
    
    def reset_ancilla(self, mappings:dict) -> str:
        ckt_str = ''
        x_measure_coords = mappings['x_measure_coords']
        z_measure_coords = mappings['z_measure_coords']
        x_measure_idx = [self.coords_index_mapper[i] for i in x_measure_coords]
        z_measure_idx = [self.coords_index_mapper[i] for i in z_measure_coords]
        reset_measure = 'R'
        ckt_str += reset_measure + ' ' + ' '.join(str(i) for i in x_measure_idx + z_measure_idx) + "\n"
        if self.init_error != None:
            ckt_str += 'X_ERROR(%f) '%(self.init_error) + ' '.join(str(i) 
                                                                   for i in x_measure_idx + z_measure_idx) + "\n"
            pass
        return ckt_str
    
    def add_gate(self, gate:str, phys_qubit:int, target_qubit:int=None) -> str:
        # Return a string specifying the gate and target qubit, and the noise annotation
        # for that gate.
        # Lookup the gate error for this phys_qubit from the noise profile
        # 1Q gates:
        noise = '\nDEPOLARIZE1(%f) %i'%(self.profile[self.index_coords_mapper[phys_qubit]]['error'][gate], phys_qubit) if self.disable_noise == False else ''
        str = gate + ' %i'%(phys_qubit) + noise + "\n"
        # 2Q gates:
        if target_qubit != None:
            noise = '\nDEPOLARIZE2(%f) %i %i'%(self.profile[self.index_coords_mapper[phys_qubit]]['error'][gate], phys_qubit, target_qubit) if self.disable_noise == False else ''
            str = gate + ' %i %i'%(phys_qubit, target_qubit) + noise + "\n"
        if gate in self.measures:
            noise = 'X_ERROR(%f) %i\n'%(self.profile[self.index_coords_mapper[phys_qubit]]['error'][gate], phys_qubit) if self.disable_noise == False else ''
            str = noise + gate + ' %i\n'%(phys_qubit)
            pass
        return str
    
    def hadamard_stage(self, coords:dict) -> str:
        ckt_str = ''
        ckt_str = self.tick(ckt_str)
        for mappings in coords.values():
            x_measure_coords = mappings['x_measure_coords']
            x_measure_idx = [self.coords_index_mapper[i] for i in x_measure_coords]
            ckt_str += ''.join(self.add_gate('H', i) for i in x_measure_idx)
        return ckt_str
    
    def cnot_stage(self, coords:dict) -> str:
        interaction_order_z = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        interaction_order_x = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        cxs = []
        str = ''
        for i in range(len(interaction_order_x)):
            ckt_str = []
            for mappings in coords.values():
                data_coords = mappings['data_coords']
                x_measure_coords = mappings['x_measure_coords']
                z_measure_coords = mappings['z_measure_coords']
                for q in x_measure_coords:
                    c = tuple((q[0] + interaction_order_x[i][0], q[1] + interaction_order_x[i][1]))
                    if c in data_coords:
                        ckt_str.append(self.add_gate('CX', self.coords_index_mapper[q], 
                                                     self.coords_index_mapper[c]))
                    pass
                for q in z_measure_coords:
                    c = tuple((q[0] + interaction_order_z[i][0], q[1] + interaction_order_z[i][1]))
                    if c in data_coords:
                        ckt_str.append(self.add_gate('CX', self.coords_index_mapper[c], 
                                                     self.coords_index_mapper[q]))
                    pass
                pass
            cxs.append(ckt_str)
        for i in range(len(interaction_order_x)):
            str = self.tick(str)
            str += ''.join(cx for cx in cxs[i])
        return str
    
    def measure_ancilla(self, coords:dict) -> str:
        ckt_str = ''
        ckt_str = self.tick(ckt_str)
        for patch, mappings in enumerate(coords.values()):
            x_measure_coords = mappings['x_measure_coords']
            z_measure_coords = mappings['z_measure_coords']
            x_measure_idx = [self.coords_index_mapper[i] for i in x_measure_coords]
            z_measure_idx = [self.coords_index_mapper[i] for i in z_measure_coords]
            measure_qubits = x_measure_idx + z_measure_idx
            ckt_str += ''.join(self.add_gate('MR', i) for i in measure_qubits)
            # Track the measurement index
            self.measurement_tracker += x_measure_coords + z_measure_coords
        return ckt_str
    
    def measure_data(self, coords:dict) -> str:
        ckt_str = ''
        ckt_str = self.tick(ckt_str)
        for mappings in coords.values():
            data_coords = mappings['data_coords']
            data_idx = [self.coords_index_mapper[i] for i in data_coords]
            self.measurement_tracker += data_coords
            ckt_str += ''.join(self.add_gate('M', i) for i in data_idx)
        return ckt_str
    
    def detector_first_round(self, mappings:dict) -> str:
        ckt_str = ''
        x_measure_coords = mappings['x_measure_coords']
        z_measure_coords = mappings['z_measure_coords']
        indices = {}
        if True: # substitute with appropriate condition for x/z basis experiment
            indices = {i:self.measurement_tracker.index(i) - 
                       len(self.measurement_tracker) 
                       for i in z_measure_coords}
        else:
            indices = {i:self.measurement_tracker.index(i) - 
                       len(x_measure_coords + z_measure_coords) 
                       for i in x_measure_coords}
        ckt_str += '\n'.join('DETECTOR(%i, %i, 0) rec[%i]'%(\
            i[0], i[1], indices[i]) for i in indices.keys()) + '\n'
        return ckt_str
    
    def add_detectors(self, mappings:dict, logical_qubits:int) -> str:
        ckt_str = ''
        x_measure_coords = mappings['x_measure_coords']
        z_measure_coords = mappings['z_measure_coords']
        indices = {i:self.measurement_tracker.index(i) - 
                   len(self.measurement_tracker) 
                   for i in x_measure_coords + z_measure_coords}
        ckt_str += '\n'.join('DETECTOR(%i, %i, 0) rec[%i] rec[%i]'%(\
            i[0], i[1], indices[i], indices[i] - logical_qubits * \
                len(x_measure_coords + z_measure_coords)) for i in indices.keys()) \
                    + '\n'
        return ckt_str
    
    def final_detectors(self, mappings:dict) -> str:
        interaction_order_z = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        data_coords = mappings['data_coords']
        z_measure_coords = mappings['z_measure_coords']
        # Last len(data_coords) measurements in measurement_tracker correspond to the data qubits
        indices = {i:self.measurement_tracker.index(i) - len(self.measurement_tracker) 
                   for i in data_coords + z_measure_coords}
        ckt_str = ''
        for coord in z_measure_coords:
            rec = ''
            for i in interaction_order_z:
                c = tuple((coord[0] + i[0], coord[1] + i[1]))
                if c in data_coords:
                    rec += 'rec[%i] '%(indices[c])
                pass
            rec += 'rec[%i]\n'%(indices[coord])
            ckt_str += 'DETECTOR(%i, %i, 1) '%(coord[0], coord[1]) + rec
        return ckt_str
    
    def add_observables(self, mappings:dict, qubit:int) -> str:
        z_observables = mappings['z_observable']
        ckt_str = ''
        # Add support for x-basis experiments
        indices = {i:self.measurement_tracker.index(i) - len(self.measurement_tracker) 
                   for i in z_observables}
        ckt_str += 'OBSERVABLE_INCLUDE(%i) '%(qubit)
        for i in z_observables:
            ckt_str += 'rec[%i] '%(indices[i])
        ckt_str += '\n'
        return ckt_str
    
    def setup_patches(self, coords:dict) -> str:
        # Take coords for all patches (logical qubits) and generate interactions
        # In case there are future qubit-qubit interactions, add the circuits for those later
        # Add only the first and d-1 rounds for the initial logical qubits. Save the 
        # last detector annotations in case there are no logical operations (memory experiment)
        self.coords_to_index_mapper(coords)

        # Specify qubit coordinates for Stim
        ckt_str = ''.join("QUBIT_COORDS(%i, %i) %i\n"%(\
            coord[0], coord[1], self.coords_index_mapper[coord]) 
                          for coord in self.coords_index_mapper.keys())
        rep_ckt = ''

        # Reset phase
        for mappings in coords.values():
            ckt_str += self.reset_data(mappings)
            ckt_str += self.reset_ancilla(mappings)

        # stim can take gates with multiple targets, but to make noise
        # annotations simpler, specify gates for each qubit in a new line. 

        # Hadamard phase
        rep_ckt += self.hadamard_stage(coords)

        # CNOT phase
        rep_ckt += self.cnot_stage(coords)

        # Hadamard phase
        rep_ckt += self.hadamard_stage(coords)

        # Measurement phase
        rep_ckt += self.measure_ancilla(coords)

        # TODO: 2 things:
        # - What if there's a logical operation on a qubit while the other qubits are free? 
        #   - How to grow the circuit for this case? (And other generic cases)
        # - How to add idling error to the final circuit? (will be a full parse of the circuit)

        # For lattice surgery, the coords of a patch will change, and so the same functions
        # above can be called for the new set of coords (Think in terms of split and merge: 
        # A split will result in a new logical with it's own set of coords, a merge will 
        # result in two logicals being fused. In the case of a merge, the first detector 
        # annotations will be slightly different since the qubits in the gap between the 
        # two logicals will be measured.)

        # Detector phase
        # For the first round, the detector just specifies the measurements that are deterministic
        ckt_str += rep_ckt
        for mappings in coords.values():
            ckt_str += self.detector_first_round(mappings)
        # ckt_str += 'REPEAT %i {\n'%(self.rounds_per_op) # REPEAT does not help wrt time -- unrolled is better
        rep_ckt += 'SHIFT_COORDS(0, 0, 1)\n'
        for mappings in coords.values():
            rep_ckt += self.add_detectors(mappings, len(list(coords.values())))
            pass
        for _ in range(self.rounds_per_op - 1): # Unroll the loop for all rounds
            ckt_str += rep_ckt
        # ckt_str += '}\n'

        # Final measurement of data qubits and detector annotations
        epilogue = ''
        epilogue += self.measure_data(coords)
        for mappings in coords.values():
            epilogue += self.final_detectors(mappings)
        
        # Add observables
        for q in coords.keys():
            # These are the default observables without any logical interactions
            epilogue += self.add_observables(coords[q], q)
            pass
        epilogue = self.tick(epilogue)
        self.default_epilogue = epilogue
        self.init_ckt = ckt_str
        return ckt_str

    def __lattice_merge(self, coords:dict) -> None:
        return
    
    def __lattice_split(self, coords:dict) -> None:
        return
    
    def add_skew(self) -> None:
        # Add skew between qubits for all gates.
        # Skew is wrt a controllers that can be considered the master
        # So some qubits will have no skew, others will have some predefined skew
        # The skew could be constant in time (not very realistic?) or vary with
        # time. Skew could also be +ve or -ve.
        # This skew would be automatically added to the idle period before a gate
        # is executed for a qubit. 
        self.skew_map = {i:0 for i in self.index_coords_mapper.keys()}
        # Skew for staggering instructions between logical patches
        stagger_time = 5
        self.stagger_patches = {i:stagger_time * self.qubit_patch_mapper[i] for i in self.index_coords_mapper.keys()}
        self.skew_map = self.stagger_patches
        # Skew between controllers
        per_logical_controller = False
        phys_qubits_per_controller = 100
        # In case a fixed number of qubits are controlled by a single controllers,
        # the comtrollers will control different square patches of qubits (rather than linear chains)
        # for coords in self.coords_index_mapper.keys():
        #     pass
        # skew introduced because of variable decoding latency
        return
    
    def add_distributed_idle_periods(self, patch:int, duration:float, k:int=1) -> None:
        self.distributed_idle = True
        self.distributed_idle_patch = patch
        self.distributed_idle_dur = duration
        self.distributed_rounds = k
        return
    
    def __get_distributed_idle(self, layers:int) -> None:
        i_c = lambda i: self.index_coords_mapper[i]
        self.distributed_idle_periods = '\n'.join(self.idling_model(self.distributed_idle_dur / (layers * self.distributed_rounds),
                                                                     i_c(i), 
                                                                     i) 
                                                  for i in self.patch_qubit_mapper[self.distributed_idle_patch])
        return
    
    def add_pre_decode_barrier(self, patch:int, duration:float) -> None:
        # Emulates the case where patches wait for all other patches to start the decode stage
        self.pre_decode = True
        i_c = lambda i: self.index_coords_mapper[i]
        self.pre_decode_idle = '\n'.join(self.idling_model(duration, i_c(i), i) 
                                          for i in self.patch_qubit_mapper[patch])
        return
    
    # Runahead might disturb the quiscent state of the lattice, probably not feasible in practice.
    def runahead_sync(self, ckt:str) -> str:
        # Create a DAG from the generated circuit for use in an appropirate 
        # scheduling pass depending on the resynchronization policy.
        ckt = np.array([line for line in ckt.split('\n')], dtype="object")
        new_ckt = ckt
        ticks = np.where(ckt == 'TICK')[0]
        # Find sub-ckt between every 2 TICKS
        intervals = np.lib.stride_tricks.sliding_window_view(ticks, 2)
        final_stage = False
        gates = {i:[] for i in self.qubit_patch_mapper.values()}
        execution_time = {}
        first_hadamard = True
        insts_conc = {}
        round_counter = 0
        
        # Get the idle periods from the DAG execution order
        def get_idle_periods(sorted_gates:list):
            # Get the idle periods from a list of tuples containing the gate, qubits
            # and the time at which that gate will be executed.
            # Every dag starts at a new frame, so the total time up to the point can
            # be simply added to the total execution time of the dag
            timesheet = {}
            insts_debug = {}
            _, last_exec_time = sorted_gates[-1]
            for gate, time in sorted_gates:
                _, *qubits, duration = gate
                for qubit in list(qubits):
                    skew = self.skew_map[int(qubit)]
                    timesheet[qubit] = [tuple((self.time + time - duration + skew, self.time + time + skew))] \
                        if qubit not in timesheet.keys() \
                        else timesheet[qubit] + [tuple((self.time + time - duration + skew, self.time + time + skew))]
                    insts_debug[self.time + time - duration + skew] = [gate] \
                        if self.time + time - duration + skew not in insts_debug.keys() \
                            else insts_debug[self.time + time - duration + skew] + [gate]
                    pass
                pass
            insts_debug = {k:list(set(insts_debug[k])) for k in insts_debug.keys()}
            idle_times = {}
            for q in timesheet.keys():
                exec_intervals = np.lib.stride_tricks.sliding_window_view(timesheet[q], 2, axis=0)
                idle_times[q] = np.sum([i[0][1] - i[1][0] + self.decode_latency
                                        for i in exec_intervals])
                pass
            # Insts that start together
            insts_conc = {time:len(insts_debug[time]) 
                          for time in insts_debug.keys()}
            nonlocal round_counter
            round_counter += 1
            addl_latency = self.decode_latency if round_counter % self.rounds_per_op == 0 else 0
            # Insts inflight at the same time
            ##
            self.time += last_exec_time + addl_latency
            return idle_times, insts_conc
        
        def create_dag(gates:list, dag:nx.DiGraph):
            _ = [dag.add_node(i, weight=i[-1]) for i in gates]
            pairs = np.fromiter(itertools.combinations(gates, 2), dtype='object')
            qubits = np.array([set(i[0][1:-1]) & set(i[1][1:-1]) for i in pairs], dtype='object')
            edge_idxs = np.where(qubits != set())[0]
            _ = [dag.add_edge(i[0], i[1]) for i in pairs[edge_idxs]]
            return dag
        
        def get_sorted_gates(g:list):
            # Inner DFS traversal function
            def dfs(node):
                max_time = 0
                for predecessor in dag.predecessors(node):
                    if predecessor in execution_time:
                        max_time = max(max_time, execution_time[predecessor])
                    else:
                        dfs(predecessor)
                        max_time = max(max_time, execution_time[predecessor])
                execution_time[node] = max_time + dag.nodes[node]['weight']
                return
            dag = nx.DiGraph()
            dag = create_dag(g, dag)
            topological_order = list(nx.topological_sort(dag))
            for node in topological_order:
                if node not in execution_time:
                    dfs(node)
            sorted_gates = sorted(execution_time.items(), key=lambda x: x[1])
            return sorted_gates

        idx_offset = 0
        frame_start = 0
        frame_end = 0
        last_annotation = False
        snip_idxs = []
        for interval_idx, interval in enumerate(intervals):
            snip = ckt[interval[0] + 1:interval[1]]
            for snip_idx, line in enumerate(snip):
                split = line.split(' ')
                gate = split[0]
                if gate not in self.gates:
                    continue
                # To prevent cycles in the graph, rename the second layer of hadamards
                if gate == 'H' and first_hadamard == False:
                    split[0] = 'H2'
                if gate != 'H' and first_hadamard == True:
                    first_hadamard = False
                if gate in ['MR', 'MX'] and first_hadamard == False:
                    frame_end = interval_idx
                    first_hadamard = True # reset
                qubit = int(split[1])
                latency = self.profile[self.index_coords_mapper[qubit]]['latency'][gate]
                split.append(latency)
                split = tuple(split)
                snip_idxs.append(snip_idx)
                if gate in self.measures and final_stage == False:
                    # This signifies the start of the end of the current DAG
                    final_stage = True
                    pass
                elif final_stage and (gate == 'H' or gate == 'M'):
                    # Measurements complete, new cycle has started
                    # Start a new DAG, complete the last one and update timers
                    final_stage = True if gate == 'M' else False
                    if last_annotation == False:
                        if round_counter >= self.rounds_per_op - self.distributed_rounds and self.distributed_idle:
                            # update skew for distributed idling
                            # This works by introducing the delay to be inserted within the last round as a constant skew
                            self.skew_map = {i:self.skew_map[i] + self.distributed_idle_dur / self.distributed_rounds 
                                             for i in self.skew_map.keys()}
                        sorted_gates = list(itertools.chain.from_iterable([get_sorted_gates(gates[k]) 
                                                                           for k in gates.keys()]))
                        # Use these sorted gates to determine idle periods for every qubit
                        idles, concurrent_insts = get_idle_periods(sorted_gates=sorted_gates)
                        layers = len(concurrent_insts.keys())
                        self.distributed_layers = layers
                        assert layers <= len(snip_idxs)
                        if self.distributed_idle:
                            self.__get_distributed_idle(2)
                        insts_conc = {**insts_conc, **concurrent_insts}
                        qubit = int(split[1])
                        self.total_idle += np.sum(list(idles.values()))
                        # divide the idling error equally between the start and end of this round
                        idle_noise = np.array([self.idling_model(idles[str(qubit)] / 2, 
                                                                self.index_coords_mapper[int(qubit)], int(qubit)) 
                                                                for qubit in idles.keys()], dtype="object")
                        if self.disable_noise == False:
                            new_ckt = np.insert(new_ckt, 
                                                intervals[frame_start][0] + 1 + idx_offset, 
                                                idle_noise)
                            idx_offset += len(idle_noise)
                            new_ckt = np.insert(new_ckt, 
                                                intervals[frame_end - 1][1] + idx_offset, 
                                                idle_noise)
                            idx_offset += len(idle_noise)
                        gates = {i:[] for i in self.qubit_patch_mapper.values()}
                        frame_start = interval_idx
                        last_annotation = True if gate == 'M' else last_annotation
                    pass
                elif gate not in self.measures:
                    # H/CX stage
                    final_stage = False
                gates[self.qubit_patch_mapper[int(split[1])]].append(split)
            pass
        # Complete timing of last set of data measurements
        if self.pre_decode and self.disable_noise == False:
            new_ckt = np.insert(new_ckt,
                                intervals[-1][0] + idx_offset + 1,
                                self.pre_decode_idle)
            idx_offset += 1
            pass
        if self.distributed_idle:
            new_ckt = np.insert(new_ckt,
                                intervals[0][0] + idx_offset + 1,
                                self.distributed_idle_periods)
            idx_offset += 1
            new_ckt = np.insert(new_ckt,
                                intervals[-1][0] + idx_offset + 1,
                                self.distributed_idle_periods)
            idx_offset += 1
            pass
        times = [gate[-1] for gate in list(gates.values())[0]]
        qubits = [int(gate[-2]) for gate in list(gates.values())[0]]
        times = np.array(times) + np.array([self.skew_map[qubit] for qubit in qubits])
        self.time += max(times)
        
        # Convert array back to string
        new_ckt = '\n'.join(i for i in new_ckt)
        self.concurrent_insts = insts_conc
        return new_ckt
    
    def fenced_sync(self, ckt:str) -> str:
        # Convert to array
        ckt = np.array([line for line in ckt.split('\n')], dtype='object')
        new_ckt = ckt
        ticks = np.where(ckt == 'TICK')[0] # Find indices of all TICKS
        # Find sub-ckt between every 2 TICKS
        intervals = np.lib.stride_tricks.sliding_window_view(ticks, 2)
        inst_cnt = {}
        idx_offset = 0
        round_counter = 0
        for interval in intervals:
            timeline = {}
            insts_debug = {}
            snip = ckt[interval[0] + 1:interval[1]]
            # This is the sub-circuit executed for all logical qubits in parallel
            max_latency = 0
            qubits = []
            skew_preamble = {}
            last_gate = None
            for inst in snip:
                split = inst.split(' ')
                gate = split[0]
                if len(split) < 2 or gate not in self.gates:
                    # Nothing useful
                    continue
                last_gate = gate
                qubit = int(split[1])
                latency = self.profile[self.index_coords_mapper[qubit]]['latency'][gate]
                skew = self.skew_map[qubit]
                # This max latency determines the duration of this TICK
                max_latency = max(max_latency, latency + skew)
                qubits.append(qubit)
                # Construct timeline for all measure qubits in the lattice
                # Will need to account for skew here
                target = None
                timeline[qubit] = [tuple((self.time, self.time + latency + skew))] \
                    if qubit not in timeline.keys() \
                    else timeline[qubit] + [tuple((self.time, self.time + latency + skew))]
                insts_debug[self.time] = [gate] if self.time not in insts_debug.keys() else insts_debug[self.time] + [gate]
                if gate in self.gates_2Q:
                    target = int(split[2])
                    qubits.append(target)
                    timeline[target] = [tuple((self.time, self.time + latency + skew))] \
                        if target not in timeline.keys() \
                        else timeline[target] + [tuple((self.time, self.time + latency + skew))]
                    pass
                # Skew contributes to idling errors before the gate, slack 
                # contributes to idling errors after a gate. 
                skew_preamble[qubit] = skew
                if target != None:
                    skew_preamble[target] = self.skew_map[target]
                pass
            if last_gate in self.measures and last_gate != 'M':
                round_counter += 1
            addl_latency = self.decode_latency if round_counter % self.rounds_per_op == 0 and round_counter > 0 and last_gate != 'M' else 0
            self.time += max_latency + addl_latency
            insts_conc = {time:len(insts_debug[time]) 
                          for time in insts_debug.keys()}
            inst_cnt = {**inst_cnt, **insts_conc}
            layers = 7 # a fixed number 7 for the fenced schedule, can't be more than this
            if self.distributed_idle and round_counter == self.rounds_per_op - self.distributed_rounds + 1:
                self.__get_distributed_idle(layers)
            # The qubits not included above will incur an idling error equal to
            # the total time of the tick
            post_time = {qubit:self.time - timeline[qubit][-1][1]
                         for qubit in qubits}
            rem = {qubit:max_latency 
                   for qubit in np.setdiff1d(self.measure_idxs + self.data_idxs,
                                             qubits)}
            post_time = {**post_time, **rem}
            self.total_idle += np.sum(list(post_time.values())) + np.sum(list(skew_preamble.values()))
            post_noise = '\n'.join(self.idling_model(post_time[qubit], 
                                                     self.index_coords_mapper[qubit], qubit) 
                                                     for qubit in post_time.keys())
            if self.disable_noise == False:
                new_ckt = np.insert(new_ckt, 
                                    interval[0] + 1 + idx_offset, 
                                    '\n'.join(self.idling_model(skew_preamble[qubit], 
                                                                self.index_coords_mapper[qubit], qubit) 
                                                                for qubit in skew_preamble.keys()))
                idx_offset += 1
                new_ckt = np.insert(new_ckt, 
                                    interval[1] + idx_offset, 
                                    post_noise)
                idx_offset += 1
                if self.distributed_idle and round_counter >= self.rounds_per_op - self.distributed_rounds + 1:
                    if round_counter == self.rounds_per_op:
                        self.distributed_idle = False
                    new_ckt = np.insert(new_ckt,
                                        interval[1] + idx_offset,
                                        self.distributed_idle_periods)
                    idx_offset += 1
                if last_gate == 'M' and self.pre_decode:
                    new_ckt = np.insert(new_ckt,
                                        interval[0] + idx_offset,
                                        self.pre_decode_idle)
                    idx_offset += 1
                pass
            pass
        new_ckt = '\n'.join(i for i in new_ckt)
        self.concurrent_insts = inst_cnt
        return new_ckt
    
    def scheduler(self, ckt:str, policy:str='runahead') -> str:
        new_ckt = ckt
        if policy == 'runahead':
            new_ckt = self.runahead_sync(ckt=ckt)
        elif policy == 'fenced':
            new_ckt = self.fenced_sync(ckt=ckt)
        else:
            raise ValueError("Only 'fenced' and 'runahead' policies are supported.")
        return new_ckt

    def synthesize(self, cmds:dict) -> None:
        # Main synthesis function. Will call other functions based on the input 
        # circuit
        # Generate initial circuit for all logical qubits.
        coords = {i:self.map_qubit(i) for i in cmds.keys()}
        ckt = self.setup_patches(coords)
        # print(ckt + self.default_epilogue)
        # Iterate through logical operations.
        return

    def get_error_rate(self, ckt:str, num_shots:int=1000000) -> float:
        num_errors, errors_per_logical = self.count_logical_errors(ckt, num_shots)
        return num_errors / num_shots, errors_per_logical / num_shots
    
    def from_string(self, ckt:str) -> None:
        cmds = circuit_parser.from_string(self, ckt)
        self.measurement_tracker = []
        self.synthesize(cmds)
        return self

    def from_file(self, ckt:str) -> None:
        cmds = circuit_parser.from_file(self, ckt)
        self.measurement_tracker = []
        self.synthesize(cmds)
        return self

    def print_stats(self) -> None:
        stat_counter.__init__(self, self.profile, self.coords_index_mapper, 
                              self.gates, self.gates_1Q, self.gates_2Q, self.measures)
        return
    
    pass

if __name__ == "__main__":
    d = 3
    sim = circuit(distance=d, num_patches_y=20, num_patches_x=20, spacing=1, disable_noise=False, fixed_t1=20, fixed_t2=30, fixed_cnot_latency=100, fixed_measure_latency=660, fixed_cnot_noise=0.0001, fixed_measure_noise=0.0001, rounds_per_op=d, idle_multiplier=1).from_string('qreg q[1];')
    # qubits = int(round(0.0588 * (2 * d**2 - 1)))
    # coords = np.array(sim.get_qubit_coords())
    # slow_errory_qubits = [tuple((2,2))]#coords[np.random.choice(len(coords), qubits), :]
    # _ = [sim.set_cnot_latency(tuple((qubit)), 400) for qubit in slow_errory_qubits]
    # ckt = sim.from_string('qreg q[1]')
    # sim.add_distributed_idle_periods(0, 1000)
    # sim.add_pre_decode_barrier(0, 1000)
    new_ckt = sim.scheduler(ckt=sim.init_ckt + sim.default_epilogue, policy='runahead')
    e, _ = sim.get_error_rate(ckt=new_ckt, num_shots=1000000)
    print(e)
    # print(new_ckt)
    # print(sim.skew_map)
    # print(sim.time, sim.total_idle)
    # print(sim.total_idle)
    # print(max(sim.concurrent_insts.values()))
    # print(sim.concurrent_insts)
    