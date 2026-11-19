# Devices (backends)

The `backends` field of a configuration takes these names.

| Value | Description |
| --- | --- |
| `custom_line`, `custom_grid`, `custom_cube`, `custom_full`, `custom_heavyhex`, `custom_diamond` | Synthetic topologies; require `backends_sizes` (or `distances`) |
| `real_flamingo`, `real_loon` | IBM Flamingo and IBM Loon/Nighthawk (multi-chip, with remote gates) |
| `real_flamingo_1_qpu`, `real_loon_1_qpu` | The same devices restricted to a single QPU |
| `real_bicycle`, `real_bicycle_lr` | IBM's bicycle architecture module: the gross code's 288 qubits in the toric layout, each with four torus couplers and two long-range c-couplers (in `_lr` as noisy as remote gates); `real_bicycle_<n>_qpus` chains n modules by remote links |
| `real_torino` | IBM Torino (Qiskit fake backend) |
| `real_willow` | Google Willow |
| `real_apollo`, `real_nsapollo` | Quantinuum Apollo with and without shuttling |
| `real_infleqtion`, `real_nsinfleqtion` | Infleqtion with and without shuttling |
| `real_harvard` | Harvard/MIT reconfigurable neutral-atom array with 448 atoms (Bluvstein et al., Nature 649, 2025): grid neighbours interact in place, all other pairs after moving an atom |
| `native`, `native_lr` | A device whose couplers are exactly the qubit pairs of the code's circuit (needs `distances`), so the code runs without SWAPs; in `native_lr` the long-range couplers of IBM's bivariate bicycle layout are as noisy as remote gates |
| `variance_<level>_<scale>` | Flamingo-like device with T1/T2 variance `<level>` ∈ {`none`, `low`, `mid`, `high`} and coherence times scaled by `<scale>`, e.g. `variance_mid_10` |
| `ibm_<device>` | A real IBM device through `QiskitRuntimeService` (requires an IBM token) |
