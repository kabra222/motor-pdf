from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

HAS_CREPES = False
try:
    from crepes import WrapClassifier, WrapRegressor
    HAS_CREPES = True
except ImportError:
    pass


class ConformalSegmenter:
    def __init__(self, confidence: float = 0.90):
        self.confidence = confidence
        self._classifier = None
        self._available = HAS_CREPES

    @property
    def is_available(self) -> bool:
        return self._available

    def calibrate(self, features: np.ndarray, boundaries: np.ndarray):
        if not self._available or features.shape[0] < 10:
            return
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000)
            clf.fit(features, boundaries)
            self._classifier = WrapClassifier(clf)
            cal_size = max(5, features.shape[0] // 3)
            self._classifier.calibrate(
                features[:cal_size],
                boundaries[:cal_size],
            )
            logger.debug("Conformal classifier calibrated with %d samples", cal_size)
        except Exception as exc:
            logger.debug("Conformal calibration failed: %s", exc)

    def predict_interval(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if not self._available or self._classifier is None:
            return features[:, 0], None
        try:
            intervals = self._classifier.predict_int(
                features, confidence=self.confidence
            )
            lower = np.array([i[0] for i in intervals])
            upper = np.array([i[1] for i in intervals])
            midpoint = (lower + upper) / 2
            return midpoint, (lower, upper)
        except Exception as exc:
            logger.debug("Conformal prediction failed: %s", exc)
            return features[:, 0], None

    def estimate_segmentation_uncertainty(
        self, segments: list, embedding_dim: int = 80
    ) -> list[dict]:
        uncertainties = []
        for i, seg in enumerate(segments):
            text = getattr(seg, "text", seg.get("text", ""))
            feat = np.zeros((1, embedding_dim), dtype=np.float32)
            words = text.lower().split()
            total = max(len(words), 1)
            feat[0, 0] = len(text) / 500.0
            feat[0, 1] = len(words) / 100.0
            feat[0, 2] = 1.0 if i > 0 else 0.0

            midpoint, interval = self.predict_interval(feat)
            if interval is not None:
                lower, upper = interval
                uncertainty = float(upper[0] - lower[0])
            else:
                uncertainty = 0.5

            uncertainties.append({
                "segment_idx": i,
                "uncertainty": round(min(1.0, uncertainty), 4),
                "confidence": round(max(0.0, 1.0 - uncertainty), 4),
            })
        return uncertainties


def wrap_regressor(model, cal_features, cal_targets):
    if not HAS_CREPES:
        return None
    try:
        wrapper = WrapRegressor(model)
        wrapper.calibrate(cal_features, cal_targets)
        return wrapper
    except Exception as exc:
        logger.debug("WrapRegressor failed: %s", exc)
        return None
