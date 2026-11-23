"""nvqldpc.fold_likely_errors (tcbp_jobs_gpu): a mechanism with p > 0.5 becomes a certain flip (syndrome and observable
biases) plus one of probability 1 - p. Runs without a GPU."""
import numpy as np
import scipy.sparse as sp

from eccentric.decoders.gpu.nvqldpc import fold_likely_errors


def test_fold_keeps_syndromes_and_observables():
    rng = np.random.default_rng(3)
    chk = sp.csr_matrix(rng.random((12, 30)) < 0.2, dtype=np.uint8)
    obs = sp.csr_matrix(rng.random((2, 30)) < 0.2, dtype=np.uint8)
    priors = rng.uniform(0.001, 0.2, 30)
    priors[[4, 17]] = [0.9, 0.999]
    folded, sb, ob = fold_likely_errors(chk, obs, priors)
    assert np.all(folded <= 0.5) and np.allclose(folded[[4, 17]], [0.1, 0.001])
    # a fault pattern e of the original mechanisms is e' = e xor (p > 0.5) of the folded ones
    for _ in range(50):
        e = (rng.random(30) < 0.5).astype(np.uint8)
        e_folded = e ^ (priors > 0.5)
        assert np.array_equal(chk @ e % 2, (chk @ e_folded + sb) % 2)
        assert np.array_equal(obs @ e % 2, (obs @ e_folded + ob) % 2)


def test_fold_is_a_no_op_without_likely_errors():
    chk = sp.csr_matrix(np.eye(3, dtype=np.uint8))
    folded, sb, ob = fold_likely_errors(chk, chk, [0.1, 0.2, 0.5])
    assert sb is None and ob is None and np.allclose(folded, [0.1, 0.2, 0.5])
