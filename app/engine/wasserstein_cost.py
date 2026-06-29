from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

HAS_RUPTURES = False
try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    pass

HAS_POT = False
try:
    import ot
    HAS_POT = True
except ImportError:
    pass

HAS_NUMPY = True


class WassersteinTopologicalCost:
    if HAS_RUPTURES and HAS_POT:

        class _CostImpl(rpt.base.BaseCost):
            model = "custom_wasserstein_1d"
            min_size = 3

            def __init__(self):
                self.signal = None
                self.n_samples = None

            def fit(self, signal):
                self.signal = signal
                self.n_samples = signal.shape[0]
                return self

            def error(self, start, end):
                if end - start < self.min_size * 2:
                    return 0.0
                sub = self.signal[start:end]
                mid = len(sub) // 2

                mag_p = np.linalg.norm(sub[:mid], axis=1)
                mag_q = np.linalg.norm(sub[mid:], axis=1)

                eps = 1e-10
                sp = np.sum(mag_p) + eps
                sq = np.sum(mag_q) + eps
                p_dist = mag_p / sp
                q_dist = mag_q / sq

                p_dist = np.maximum(p_dist, eps)
                q_dist = np.maximum(q_dist, eps)
                p_dist /= p_dist.sum()
                q_dist /= q_dist.sum()

                cp = np.arange(len(p_dist), dtype=np.float64) / max(len(p_dist), 1)
                cq = np.arange(len(q_dist), dtype=np.float64) / max(len(q_dist), 1)

                try:
                    w1 = ot.wasserstein_1d(cp, cq, p_dist, q_dist, p=1)
                except Exception:
                    w1 = float(np.abs(np.mean(mag_p) - np.mean(mag_q)))

                scale = (end - start) / max(self.n_samples, 1)
                return float(w1 * (1.0 + scale * 10.0))

        implementation = _CostImpl
    else:
        implementation = None

    @classmethod
    def create(cls):
        if cls.implementation is not None:
            return cls.implementation()
        return None

    @classmethod
    def is_available(cls) -> bool:
        return HAS_RUPTURES and HAS_POT
