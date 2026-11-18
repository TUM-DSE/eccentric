"""CUDA plumbing for the GPU decoders: CuPy is imported on first use, so that ECCentric runs without it on
machines without a GPU; kernels are compiled with NVRTC on first use (CuPy caches the binaries)."""
import functools
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# Kernels are compiled without fused multiply-adds, so that their double arithmetic rounds like the CPU code's
NVRTC_OPTIONS = ("--fmad=false", "-std=c++17")
# Device memory that one decoder's workspaces and per-chunk buffers may take
MEMORY_BUDGET = int(os.environ.get("ECCENTRIC_GPU_MEMORY", 16 << 30))


def _preload_pip_cuda_libraries():
    """CuPy finds NVRTC through the dynamic loader; the nvidia-cuda-nvrtc-cu12 wheel is not on its path, so load
    it (and its builtins) here: later dlopen calls by soname then find it already loaded"""
    import ctypes
    import glob
    try:
        import nvidia.cuda_nvrtc as nvrtc
    except ImportError:
        return
    for directory in getattr(nvrtc, "__path__", []):
        for pattern in ("libnvrtc-builtins.so*", "libnvrtc.so*"):
            for path in sorted(glob.glob(os.path.join(directory, "lib", pattern))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


@functools.lru_cache(maxsize=None)
def cupy():
    _preload_pip_cuda_libraries()
    try:
        import cupy as cp
    except ImportError as e:
        raise RuntimeError("the GPU decoders need CuPy (pip install cupy-cuda12x) and an NVIDIA GPU") from e
    return cp


@functools.lru_cache(maxsize=None)
def module(name: str):
    """The compiled CUDA source eccentric/decoders/gpu/<name>.cu"""
    cp = cupy()
    with open(os.path.join(HERE, f"{name}.cu")) as f:
        return cp.RawModule(code=f.read(), options=NVRTC_OPTIONS)


@functools.lru_cache(maxsize=None)
def device_properties():
    cp = cupy()
    return cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)


def multiprocessors() -> int:
    return device_properties()["multiProcessorCount"]


def max_shared_per_block() -> int:
    """Dynamic shared memory a block may opt in to"""
    props = device_properties()
    return int(props.get("sharedMemPerBlockOptin", props["sharedMemPerBlock"]))


def threads_for(width: int, cap: int = 256) -> int:
    """Threads per block for a kernel whose loops run over `width` rows or columns"""
    return int(min(cap, max(32, -(-width // 32) * 32)))
