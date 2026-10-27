"""Slices of the configuration space that the sweep covers, one family of codes at a time.

A configuration is a code (with a schedule and distance), a device with a noise model (and error
rate), a compiler setting and a number of syndrome rounds; each configuration is decoded with
several decoders on the same shots.
"""
from eccentric.codes.catalog import CODES, CodeFamily, FixedCode

QLDPC_PREFIXES = ("bb", "bpc", "gb", "hgp", "lp", "lcs", "qt", "mitten", "qlp")
QLDPC_CODES = sorted(c for c in CODES if c.startswith(QLDPC_PREFIXES)) + ["gross"]

# Paper codes and their distances in the reference slice
PAPER_DISTANCES = {"surface": [3, 5, 7], "color": [3, 5, 7], "hh": [3, 5, 7], "bacon": [3, 5, 7], "steane": [3, 9], "gross": [12]}
MATCHABLE = {"surface", "hh", "bacon", "repetition", "xzzx", "toric", "honeycomb", "planar_honeycomb"}

DECODE = dict(target_errors=100, max_shots=100_000, time_budget=3600, first_batch=100, max_batch=5000, timeout=3 * 3600)


def schedules(code):
    return (CODES[code].schedules or (None,)) if code in CODES else (None,)


def code_distances(code):
    if code in PAPER_DISTANCES:
        return PAPER_DISTANCES[code]
    spec = CODES[code]
    if isinstance(spec, FixedCode):
        return [spec.distance]
    return [spec.min_d + i * spec.step for i in range(3)]


def qldpc_ibm():
    """qLDPC codes on IBM devices: Heron, Loon/Nighthawk (one or two QPUs) and Flamingo (one or three
    QPUs), at the devices' noise and, for Loon and Flamingo, tenfold lower noise"""
    ibm = ["real_flamingo", "real_flamingo_downscaled"]
    devices = [("real_torino", ["real_heron"]), ("real_loon_1_qpu", ibm), ("real_loon", ibm),
               ("real_flamingo_1_qpu", ibm), ("real_flamingo", ibm)]
    for code in QLDPC_CODES:
        for schedule in schedules(code):
            for d in code_distances(code):
                for backend, noises in devices:
                    for error_type in noises:
                        yield dict(code=code, schedule=schedule, distance=d, cycles=d, backend=backend,
                                   error_type=error_type, error_probability=None,
                                   decoders=["bposd", "bplsd", "relay_bp", "belief_find", "tesseract"])


def qldpc_ibm_projected(max_qubits=600):
    """qLDPC codes that fit none of the IBM devices above, on projected Loon and Flamingo devices with
    as many QPUs in the chain as the code needs, at tenfold lower noise (at the devices' own noise
    no qLDPC code works even on the existing devices). Tesseract is left out, it ran out of time on
    most codes above 200 qubits."""
    from eccentric.codes import get_min_n

    devices = [("loon", 121, 242), ("flamingo", 133, 399)]
    for code in QLDPC_CODES:
        for d in code_distances(code):
            n = get_min_n(code, d)
            for device, qpu_size, largest in devices:
                if n <= largest or n > max_qubits:
                    continue
                for schedule in schedules(code):
                    yield dict(code=code, schedule=schedule, distance=d, cycles=d,
                               backend=f"real_{device}_{-(-n // qpu_size)}_qpus", error_type="real_flamingo_downscaled",
                               error_probability=None, decoders=["bposd", "relay_bp", "belief_find"])


def reference(max_qubits=600):
    """Every code at its default schedule on an all-to-all device with SD6 noise, the common ground
    that makes the device slices comparable"""
    from eccentric.codes import get_min_n

    for code in list(CODES) + list(PAPER_DISTANCES):
        for d in code_distances(code):
            n = get_min_n(code, d)
            if n > max_qubits:
                continue
            decoders = ["bposd"]
            if code in MATCHABLE or code.startswith(("hyperbolic", "semi_hyperbolic")):
                decoders.append("mwpm")
            if code.startswith(QLDPC_PREFIXES) or code == "gross":
                decoders.append("relay_bp")
            if code == "color":
                decoders.append("chromobius")
            for p in (5e-4, 1e-3, 2e-3, 4e-3):
                yield dict(code=code, schedule=None, distance=d, cycles=d, backend="custom_full", backend_size=n,
                           error_type="sd6", error_probability=p, decoders=decoders)


SLICES = {"qldpc_ibm": qldpc_ibm, "qldpc_ibm_projected": qldpc_ibm_projected, "reference": reference}
