# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from typing import Union, List, Dict, Callable

from stim import Circuit as StimCircuit
from stim import target_rec as StimTarget_rec

from qiskit import transpile, QuantumCircuit
# from qiskit_aer.noise.errors.quantum_error import QuantumChannelInstruction
from qiskit_aer.noise.errors.quantum_error import QuantumChannel as QuantumChannelInstruction
from qiskit_aer.noise import pauli_error
from qiskit.circuit.library import XGate, RZGate
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling

import surface_code_decoder_v2

def transDD(circ, backend, echo="X", echo_num=2, qubit_list=[]):
    """
    Args:
        circ: QuantumCircuit object
        backend (qiskit.providers.ibmq.IBMQBackend): Backend to transpile and schedule the
        circuits for. The numbering of the qubits in this backend should correspond to
        the numbering used in `self.links`.
        echo: gate sequence (expressed as a string) to be used on the qubits. Valid strings 
        are `'X'` and `'XZX'`.
        echo_num: Number of times to repeat the sequence for qubits.
    Returns:
        transpiled_circuit: As `self.circuit`, but with the circuits scheduled, transpiled and
        with dynamical decoupling added.
    """

    initial_layout = []
    initial_layout += [
        circ.qubits[q] for q in range(circ.num_qubits)
    ]

    # transpile to backend and schedule
    circuit = transpile(
        circ, backend, initial_layout=initial_layout,scheduling_method="alap"
    )

    #then dynamical decoupling if needed
    if echo_num:

        # set up the dd sequences
        spacings = []
        if echo == "X":
            dd_sequences = [XGate()] * echo_num
            spacings.append(None)
        elif echo == "XZX":
            dd_sequences = [XGate(), RZGate(np.pi), XGate()] * echo_num
            d = 1.0 / (2 * echo_num - 1 + 1)
            spacing = [d / 2] + ([0, d, d] * echo_num)[:-1] + [d / 2]
            for _ in range(2):
                spacing[0] += 1 - sum(spacing)
            spacings.append(spacing)
        else:
            dd_sequences.append(None)
            spacings.append(None)

        # add in the dd sequences
        durations = InstructionDurations().from_backend(backend)
        if dd_sequences[0]:
            if echo_num:
                if qubit_list == []:
                    qubits = circ.qubits
                else:
                    qubits = qubit_list
            else:
                qubits = None
            pm = PassManager([ALAPScheduleAnalysis(durations),
                  PadDynamicalDecoupling(durations, dd_sequences,qubits=qubits)])
            circuit = pm.run(circuit)

        # make sure delays are a multiple of 16 samples, while keeping the barriers
        # as aligned as possible
        total_delay = [{q: 0 for q in circuit.qubits} for _ in range(2)]
        for gate in circuit.data:
            if gate[0].name == "delay":
                q = gate[1][0]
                t = gate[0].params[0]
                total_delay[0][q] += t
                new_t = 16 * np.ceil((total_delay[0][q] - total_delay[1][q]) / 16)
                total_delay[1][q] += new_t
                gate[0].params[0] = new_t

        # transpile to backend and schedule again
        # circuit = transpile(circuit, backend, scheduling_method="alap")

    return circuit

def get_stim_circuits(
    circuit: Union[QuantumCircuit, List],
    detectors: List[Dict] = None,
    logicals: List[Dict] = None,
):
    """Converts compatible qiskit circuits to stim circuits.
       Dictionaries are not complete. For the stim definitions see:
       https://github.com/quantumlib/Stim/blob/main/doc/gates.md
    Args:
        circuit: Compatible gates are Paulis, controlled Paulis, h, s,
        and sdg, swap, reset, measure and barrier. Compatible noise operators
        correspond to a single or two qubit pauli channel.
        detectors: A list of measurement comparisons. A measurement comparison
        (detector) is either a list of measurements given by a the name and index
        of the classical bit or a list of dictionaries, with a mandatory clbits
        key containing the classical bits. A dictionary can contain keys like
        'qubits', 'time', 'basis' etc.
        logicals: A list of logical measurements. A logical measurement is a
        list of classical bits whose total parity is the logical eigenvalue.
        Again it can be a list of dictionaries.

    Returns:
        stim_circuits, stim_measurement_data
    """

    if detectors is None:
        detectors = [{}]
    if logicals is None:
        logicals = [{}]

    if len(detectors) > 0 and isinstance(detectors[0], List):
        detectors = [{"clbits": det, "qubits": [], "time": 0} for det in detectors]

    if len(logicals) > 0 and isinstance(logicals[0], List):
        logicals = [{"clbits": log} for log in logicals]

    stim_circuits = []
    stim_measurement_data = []
    if isinstance(circuit, QuantumCircuit):
        circuit = [circuit]
    for circ in circuit:
        stim_circuit = StimCircuit()

        qiskit_to_stim_dict = {
            "id": "I",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "h": "H",
            "s": "S",
            "sdg": "S_DAG",
            "cx": "CX",
            "cy": "CY",
            "cz": "CZ",
            "swap": "SWAP",
            "reset": "R",
            "measure": "M",
            "barrier": "TICK",
        }
        pauli_error_1_stim_order = {
            "id": 0,
            "I": 0,
            "X": 1,
            "x": 1,
            "Y": 2,
            "y": 2,
            "Z": 3,
            "z": 3,
        }
        pauli_error_2_stim_order = {
            "II": 0,
            "IX": 1,
            "IY": 2,
            "IZ": 3,
            "XI": 4,
            "XX": 5,
            "XY": 6,
            "XZ": 7,
            "YI": 8,
            "YX": 9,
            "YY": 10,
            "YZ": 11,
            "ZI": 12,
            "ZX": 13,
            "ZY": 14,
            "ZZ": 15,
        }

        measurement_data = []
        qreg_offset = {}
        creg_offset = {}
        prevq_offset = 0
        prevc_offset = 0
        for inst, qargs, cargs in circ.data:
            for qubit in qargs:
                if qubit._register.name not in qreg_offset:
                    qreg_offset[qubit._register.name] = prevq_offset
                    prevq_offset += qubit._register.size
            for bit in cargs:
                if bit._register.name not in creg_offset:
                    creg_offset[bit._register.name] = prevc_offset
                    prevc_offset += bit._register.size

            qubit_indices = [
                qargs[i]._index + qreg_offset[qargs[i]._register.name] for i in range(len(qargs))
            ]

            if isinstance(inst, QuantumChannelInstruction):
                qerror = inst._quantum_error
                pauli_errors_types = qerror.to_dict()["instructions"]
                pauli_probs = qerror.to_dict()["probabilities"]
                if pauli_errors_types[0][0]["name"] in pauli_error_1_stim_order:
                    probs = 4 * [0.0]
                    for pind, ptype in enumerate(pauli_errors_types):
                        probs[pauli_error_1_stim_order[ptype[0]["name"]]] = pauli_probs[pind]
                    stim_circuit.append("PAULI_CHANNEL_1", qubit_indices, probs[1:])
                elif pauli_errors_types[0][0]["params"][0] in pauli_error_2_stim_order:
                    # here the name is always 'pauli' and the params gives the Pauli type
                    probs = 16 * [0.0]
                    for pind, ptype in enumerate(pauli_errors_types):
                        probs[pauli_error_2_stim_order[ptype[0]["params"][0]]] = pauli_probs[pind]
                    stim_circuit.append("PAULI_CHANNEL_2", qubit_indices, probs[1:])
                else:
                    raise Exception("Unexpected operations: " + str([inst, qargs, cargs]))
            else:
                # Gates and measurements
                if inst.name in qiskit_to_stim_dict:
                    if len(cargs) > 0:  # keeping track of measurement indices in stim
                        measurement_data.append([cargs[0]._register.name, cargs[0]._index])

                    if qiskit_to_stim_dict[inst.name] == "TICK":  # barrier
                        stim_circuit.append("TICK")
                    elif inst.condition is not None:  # handle c_ifs
                        if inst.name in "xyz":
                            if inst.condition[1] == 1:
                                clbit = inst.condition[0]
                                stim_circuit.append(
                                    qiskit_to_stim_dict["c" + inst.name],
                                    [
                                        StimTarget_rec(
                                            measurement_data.index(
                                                [clbit._register.name, clbit._index]
                                            )
                                            - len(measurement_data)
                                        ),
                                        qubit_indices[0],
                                    ],
                                )
                            else:
                                raise Exception(
                                    "Classically controlled gate must be conditioned on bit value 1"
                                )
                        else:
                            raise Exception(
                                "Classically controlled " + inst.name + " gate is not supported"
                            )
                    else:  # gates/measurements acting on qubits
                        stim_circuit.append(qiskit_to_stim_dict[inst.name], qubit_indices)
                else:
                    raise Exception("Unexpected operations: " + str([inst, qargs, cargs]))

        if detectors != [{}]:
            for det in detectors:
                stim_record_targets = []
                for reg, ind in det["clbits"]:
                    stim_record_targets.append(
                        StimTarget_rec(measurement_data.index([reg, ind]) - len(measurement_data))
                    )
                if det["time"] != []:
                    stim_circuit.append(
                        "DETECTOR", stim_record_targets, det["qubits"] + [det["time"]]
                    )
                else:
                    stim_circuit.append("DETECTOR", stim_record_targets, [])
        if logicals != [{}]:
            for log_ind, log in enumerate(logicals):
                stim_record_targets = []
                for reg, ind in log["clbits"]:
                    stim_record_targets.append(
                        StimTarget_rec(measurement_data.index([reg, ind]) - len(measurement_data))
                    )
                stim_circuit.append("OBSERVABLE_INCLUDE", stim_record_targets, log_ind)

        stim_circuits.append(stim_circuit)
        stim_measurement_data.append(measurement_data)

    return stim_circuits, stim_measurement_data

def build_stim_circuit(code, simplify = False, return_relevant_nodes = False):
    def get_coupling_map(code):
        heavyHEX_dict = code.heavyHEX_dict
        heavyHEX_inv_dict = {v:k for k,v in heavyHEX_dict.items()}
        coupling_map = []
        for c_list, t_list in code.pair_target_pos_dict.values():
            for c,t in zip(c_list,t_list):
                if (c+t)/2 in heavyHEX_dict:
                    m = (c+t)/2
                else:
                    m = [(c+t+r)/2 for r in [1+1j,1-1j,-1+1j,-1-1j] 
                                if (c+t+r)/2 in heavyHEX_dict and
                                (c+t+r)/2 not in [c,t]
                                ][0]
                coupling_map.append([heavyHEX_dict[c],heavyHEX_dict[m]])
                coupling_map.append([heavyHEX_dict[m],heavyHEX_dict[c]])
                coupling_map.append([heavyHEX_dict[m],heavyHEX_dict[t]])
                coupling_map.append([heavyHEX_dict[t],heavyHEX_dict[m]])
        for anc in code.LS_ancillas:
            anc_pos = heavyHEX_inv_dict[anc]
            if anc_pos+1 in heavyHEX_dict:
                coupling_map.append([heavyHEX_dict[anc_pos],heavyHEX_dict[anc_pos+1]])
                coupling_map.append([heavyHEX_dict[anc_pos+1],heavyHEX_dict[anc_pos]])
            if anc_pos-1 in heavyHEX_dict:
                coupling_map.append([heavyHEX_dict[anc_pos],heavyHEX_dict[anc_pos-1]])
                coupling_map.append([heavyHEX_dict[anc_pos-1],heavyHEX_dict[anc_pos]])
        return coupling_map

    def extract_error_rates(circuit):
        i = 0
        CXerror = 0
        Rerror = 0
        singleQerror = 0
        while (CXerror == 0 or Rerror == 0 or singleQerror == 0) and i<len(circuit.data):
            inst = circuit.data[i][0]
            if isinstance(inst,QuantumChannelInstruction):
                if inst.num_qubits==2:
                    CXerror = sum(inst._quantum_error.to_dict()['probabilities'][1:])
                elif inst.num_qubits==1:
                    if len(inst._quantum_error.to_dict()['instructions'])>2:
                        singleQerror = sum(inst._quantum_error.to_dict()['probabilities'][1:])
                    else:
                        Rerror = sum(inst._quantum_error.to_dict()['probabilities'][1:])
            i+=1
        return CXerror,Rerror,singleQerror

    qc = code.circuit
    if simplify:
        coupling_map = get_coupling_map(code)
        CXerror,Rerror,singleQerror = extract_error_rates(qc)
        qc_noiseless = QuantumCircuit()
        for qreg in qc.qregs:
            qc_noiseless.add_register(qreg)
        for creg in qc.cregs:
            qc_noiseless.add_register(creg)
        for gate in qc:
            if gate.operation.name!='quantum_channel':
                qc_noiseless.append(gate)
        qc_trans = transpile(qc_noiseless, optimization_level=2, coupling_map=coupling_map)
        qc = noisify_circuit(qc_trans, {'p1Q':singleQerror,'p2Q':CXerror,'pRR':Rerror,'pid':0})
    stim_circuit = get_stim_circuits(qc)[0][0]

    ese = code.error_sensitive_events.copy()
    meas_data = code.measuredict

    ind_to_coord = {v:k for k,v in meas_data.items()}
    det_coord_list = []
    Xdet_inds = []
    Zdet_inds = []
    det_inds3CX = []
    det_indsBS = []
    for i,detector in enumerate(ese):
        det_coord = [ind_to_coord[meas_ind] for meas_ind in detector]
        time = max([ind_to_coord[det][1] for det in detector])
        if time==0 and code.log_gate != ['']*len(code.log_gate):
            time-=code.T
        elif time==code.T and code.log_gate != ['']*len(code.log_gate):
            time+=code.T
        if all([m_coord[0].real%2 for m_coord in det_coord]):
            #BS
            x_offset = 4*code.d
            coords = (sum([ind_to_coord[measind][0] for measind in detector])/len(detector))
            det_coords = (coords.real+x_offset,coords.imag,time)
            det_indsBS.append(i)
        elif not any([m_coord[0].real%2 for m_coord in det_coord]):
            #3CX        
            x_offset = 0
            if ind_to_coord[detector[-1]][1]==code.T:
                det_coords = [int(ind_to_coord[detector[0]][0].real),
                            int(ind_to_coord[detector[0]][0].imag),
                            time]
                if ind_to_coord[detector[0]][-1]==code.T:
                    det_coords[0]+=2
                    det_coords[1]+=2
            else:
                det_coords = [int(ind_to_coord[detector[-1]][0].real),
                            int(ind_to_coord[detector[-1]][0].imag),
                            time]
            det_inds3CX.append(i)
        else:
            #merged dets
            x_offset = 4*code.d
            coord_list = [coord for coord,_ in det_coord if coord.real%2]
            coords = sum(coord_list)/len(coord_list)
            det_coords = (coords.real+x_offset,coords.imag,time)

        stim_circuit.append("DETECTOR",[StimTarget_rec(measind-len(meas_data)) for measind in detector],det_coords)
        det_coord_list.append(list(det_coords))

        if ((det_coord[0][0] in code.measure_z_qubit_posBS)
            or (det_coord[0][1]%2==0 and det_coord[0][0] in code.measure_z_qubit_pos_cycle1)
            or (det_coord[0][1]%2==1 and det_coord[0][0] in code.measure_z_qubit_pos_cycle0)):
            Zdet_inds.append(i)
        else:
            Xdet_inds.append(i)


    if 'LS' not in code.log_gate[-1]:
        relevant_nodes = []
        if det_inds3CX != []:
            stim_circuit.append("OBSERVABLE_INCLUDE", [StimTarget_rec(meas_data[(pos,code.T)]-len(meas_data)) for pos in code.edge_qubit_pos3CX], 0)
            if code.logical_meas[0] == 'X':
                relevant_nodes.append([ind for ind in Xdet_inds if ind not in det_indsBS])
            else:
                relevant_nodes.append([ind for ind in Zdet_inds if ind not in det_indsBS])
        if det_indsBS != []:
            stim_circuit.append("OBSERVABLE_INCLUDE", [StimTarget_rec(meas_data[(pos,code.T)]-len(meas_data)) for pos in code.edge_qubit_posBS], not det_inds3CX == [])
            if code.logical_meas[1] == 'X':
                relevant_nodes.append([ind for ind in Xdet_inds if ind not in det_inds3CX])
            else:
                relevant_nodes.append([ind for ind in Zdet_inds if ind not in det_inds3CX])
    else:
        stim_circuit.append("DETECTOR", [StimTarget_rec(meas_data[(pos,code.T)]-len(meas_data)) for pos in code.LSXX_pos + code.edge_qubit_pos3CX],[0,0,2*code.T])
        det_coord_list.append([0,0,2*code.T])
        stim_circuit.append("DETECTOR", [StimTarget_rec(meas_data[(pos,code.T)]-len(meas_data)) for pos in code.LSZZ_pos + code.edge_qubit_posBS],[0,1,2*code.T])
        det_coord_list.append([0,1,2*code.T])
        Zdet_inds.append(len(ese))
        Xdet_inds.append(len(ese)+1)

        stim_circuit.append("OBSERVABLE_INCLUDE", [StimTarget_rec(meas_data[(pos,code.T)]-len(meas_data)) for pos in code.LSXX_pos],0)
        stim_circuit.append("OBSERVABLE_INCLUDE", [StimTarget_rec(meas_data[(pos,code.T)]-len(meas_data)) for pos in code.LSZZ_pos],1)
        
        relevant_nodes = [Xdet_inds, Zdet_inds]

    if return_relevant_nodes:
        return stim_circuit, relevant_nodes
    else:
        return stim_circuit
    
def string2detections(job_result, det_measinds, log_measinds):
    meas_samples = []
    freqs = []
    for string,freq in job_result.items():
        meas_sample = np.array([int(char) for char in string[::-1]])
        meas_samples.append(meas_sample)
        freqs.append(freq)
    
    detector_samples = []
    for sample in meas_samples:
        new_det_sample = []
        for detector in det_measinds:
            new_det_sample.append(sample[detector].sum()%2)
        for log_measind in log_measinds:
            new_det_sample.append(sample[log_measind].sum()%2)
        detector_samples.append(new_det_sample)
    detector_samples = np.array(detector_samples)
    
    return freqs,detector_samples

def noisify_circuit(circuit: QuantumCircuit, error_dict: dict):
    """
    Inserts error operations into a circuit according to a pauli noise model.
    Handles idling errors in the form of custom gates "idle_#" which are assumed to
    encode the identity gate only.
    qc = QuantumCircuit(1, name='idle_1')
    qc.i(0)
    idle_1 = qc.to_instruction()

    Args:
        circuits: Circuit or list thereof to which noise is added.
        noise_model: noise model as a dictionary.

    Returns:
        noisy_circuits: Corresponding circuit or list thereof.
    """

    p1Q = error_dict['p1Q']
    p2Q = error_dict['p2Q']
    pid = error_dict['pid']
    pRR = error_dict['pRR']

    noise_model = {
                    'reset': {
                        "chan": {
                                    'i':1-pRR,
                                    'x':pRR
                                }
                            },
                    'measure': {
                        "chan": {
                                    'i':1-pRR,
                                    'x':pRR
                                }
                            },
                    'h': {
                        "chan": {
                                    'i':1-p1Q
                                }|
                                {
                                    i:p1Q/3
                                    for i in 'xyz'
                                }
                            },
                    's': {
                        "chan": {
                                    'i':1-p1Q
                                }|
                                {
                                    i:p1Q/3
                                    for i in 'xyz'
                                }
                            },
                    'sdg': {
                        "chan": {
                                    'i':1-p1Q
                                }|
                                {
                                    i:p1Q/3
                                    for i in 'xyz'
                                }
                            },
                    #idling error attached to a custom gate "idle_1" (assumed to be the identity acting on a single qubit) 
                    'idle_1': {
                        "chan": {
                                    'i':1-pid
                                }|
                                {
                                    i:pid/3
                                    for i in 'xyz'
                                }
                            },
                    'cx': {
                        "chan": {
                                    'ii':1-p2Q
                                }|
                                {
                                    i+j:p2Q/15
                                    for i in 'ixyz' 
                                    for j in 'ixyz' 
                                    if i+j!='ii'
                                }
                            }
                }
    

    # create pauli errors for all errors in noise model
    errors = {}
    for g, noise in noise_model.items():
        paulis = [pauli.upper() for pauli in noise["chan"].keys()]
        probs = list(noise["chan"].values())
        errors[g] = pauli_error(list(zip(paulis, probs)))

    qc = circuit
    noisy_qc = QuantumCircuit()
    for qreg in qc.qregs:
        noisy_qc.add_register(qreg)
    for creg in qc.cregs:
        noisy_qc.add_register(creg)

    for gate in qc:
        g = gate[0].name
        qubits = gate[1]
        pre_error = g == "reset"
        # add gate if it needs to go before the error
        if pre_error:
            noisy_qc.append(gate)
        # then the error
        if g in errors:
            noisy_qc.append(errors[g], qubits)
        # add gate if it needs to go after the error
        if not pre_error:
            if not g.startswith("idle_"):
                noisy_qc.append(gate)


    return noisy_qc

def Fidelity_yield_from_detector_samples(code,freqs, det_samples, post_sel=True, get_YY=False):
    "Calculates logical Fidelity and the error thereof. Works for individual logicals as well as for the Bell-state, when XX and ZZ are measured simultaneously."
    detector_samples = []
    for freq,sample in zip(freqs,det_samples):
        for _ in range(freq):
            detector_samples.append(sample)
    detector_samples = np.array(detector_samples)
    num_shots = len(detector_samples)

    stim_circuit,relevant_nodes = build_stim_circuit(code,return_relevant_nodes=True)
    stim_DEM = stim_circuit.detector_error_model(decompose_errors=True,approximate_disjoint_errors=True,ignore_decomposition_failures=True)
    
    if post_sel:
        matching = surface_code_decoder_v2.DEM_to_postsel_matching(model=stim_DEM, relevant_nodes=relevant_nodes)
    else:
        matching = surface_code_decoder_v2.DEM_to_matching(model=stim_DEM, relevant_nodes=relevant_nodes)

    correction_batch = matching.decode_batch(detector_samples)
    num_shots2=len(correction_batch)
    num_fail=sum(correction_batch)

    if len(num_fail) == 1:
        Fid = min(num_shots2-num_fail)/num_shots2
    elif get_YY:
        Fid = np.array([sum(correction_batch)[0], sum([sum(corr)%2 for corr in correction_batch]), sum(correction_batch)[1]])/num_shots2 ## [XX, -YY, ZZ]
    else:
        Fid = (num_shots2-num_fail)/num_shots2

    print('d =',code.d,' T =',code.T,' F =',np.round(Fid,4),'+/-',np.round(2*np.sqrt(min(num_fail))/num_shots2,4),' yield =',num_shots2/num_shots)

    return [code.d,code.T,Fid, 2*np.sqrt(min(num_fail))/num_shots, num_shots2/num_shots]