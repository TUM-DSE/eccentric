"""GPU versions of ECCentric's decoders (see analysis/GPU.md). Importing this package needs no GPU; each decoder imports
CuPy (and CUDA-Q QEC for bposd_gpu and bposd_tuned_gpu) when first called."""

# decoder name -> (module, function); decode.decode dispatches on these
GPU_DECODERS = {
    "bposd_gpu": ("nvqldpc", "bposd_gpu"),
    "bposd_tuned_gpu": ("nvqldpc", "bposd_gpu"),
    "relay_bp_gpu": ("relay_gpu", "relay_bp_gpu"),
    "bplsd_gpu": ("ldpc_gpu", "bplsd_gpu"),
    "belief_find_gpu": ("ldpc_gpu", "belief_find_gpu"),
    "hier_bposd_gpu": ("ldpc_gpu", "hierarchical_gpu"),
    "hier_bposd_hard_gpu": ("ldpc_gpu", "hierarchical_gpu"),
    "union_find_gpu": ("union_find_gpu", "union_find_gpu"),
    "gari_gpu": ("gari_gpu", "gari_gpu"),
    "tcbp_jobs_gpu": ("nvqldpc", "tcbp_jobs_gpu"),
    "tcbp_jobs_sc_gpu": ("nvqldpc", "tcbp_jobs_sc_gpu"),
}


def gpu_decoder(name: str):
    import importlib
    module, function = GPU_DECODERS[name]
    return getattr(importlib.import_module(f".{module}", __name__), function)
