"""Plugins used by test_plugins.py, registered there through fake entry points"""
import stim
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap

from eccentric.codes.catalog import CodeFamily
from eccentric.decoders.mwpm import mwpm
from eccentric.noise.artificial_noise import ArtificialNoise


def _rotated_surface(d, rounds, schedule, basis):
    return stim.Circuit.generated(f"surface_code:rotated_memory_{basis.lower()}", distance=d, rounds=rounds)


plugin_surface = CodeFamily(_rotated_surface, lambda d: 2 * d * d - 1)
# A plugin may not replace a paper code
surface = CodeFamily(_rotated_surface, lambda d: 1)

decoded = []


def plugin_decoder(circuit, num_shots, approximate_disjoint_errors):
    decoded.append(num_shots)
    return mwpm(circuit, num_shots, approximate_disjoint_errors)


def plugin_ring(backend_size):
    return GenericBackendV2(backend_size, coupling_map=CouplingMap.from_ring(backend_size))


def plugin_noise(qt, p, backend):
    return ArtificialNoise.SD6(p or 0.001, qt)


def plugin_command(argv):
    print("plugin command", *argv)
    return 0
