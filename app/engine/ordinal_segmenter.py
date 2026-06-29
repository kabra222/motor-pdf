from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    pass

HAS_SPACECUTTER = False
try:
    HAS_SPACECUTTER = True
except ImportError:
    pass


class OrdinalHierarchicalSegmenter:
    def __init__(self, embedding_dim: int = 80, max_levels: int = 4):
        self.embedding_dim = embedding_dim
        self.max_levels = max_levels
        self._available = HAS_TORCH and HAS_SPACECUTTER
        self._model = None

    @property
    def is_available(self) -> bool:
        return self._available

    def _build_model(self):
        if not self._available:
            return
        try:
            from spacecutter.losses import CumulativeLinkLoss
            from spacecutter.models import OrdinalLogisticModel
        except ImportError:
            self._available = False
            return

        class _Segmenter(nn.Module):
            def __init__(self, dim: int, levels: int):
                super().__init__()
                self.projector = nn.Sequential(
                    nn.Linear(dim, max(dim // 2, 4)),
                    nn.GELU(),
                    nn.Dropout(0.15),
                    nn.Linear(max(dim // 2, 4), 1),
                )
                self.head = OrdinalLogisticModel(self.projector, levels)
                self.criterion = CumulativeLinkLoss()

            def forward(self, x):
                return self.head(x)

            def loss(self, pred, target):
                return self.criterion(pred, target)

        self._model = _Segmenter(self.embedding_dim, self.max_levels)

    def predict_depth(self, features):
        if not self._available or self._model is None:
            self._build_model()
        if self._model is None:
            return [0] * features.shape[0] if hasattr(features, "shape") else [0]
        try:
            with torch.no_grad():
                x = torch.from_numpy(features).float() if isinstance(features, np.ndarray) else torch.tensor(features, dtype=torch.float)
                logits = self._model(x)
                preds = torch.argmax(logits, dim=1).numpy()
                return preds.tolist()
        except Exception as exc:
            logger.debug("Ordinal prediction failed: %s", exc)
            return [0] * features.shape[0] if hasattr(features, "shape") else [0]

    def classify_segments(self, segments: list, block_features: list | None = None):
        if not segments:
            return segments
        if block_features is not None:
            depths = self.predict_depth(np.array(block_features, dtype=np.float32))
        else:
            feat = np.zeros((len(segments), self.embedding_dim), dtype=np.float32)
            for i, seg in enumerate(segments):
                text = getattr(seg, "text", seg.get("text", ""))
                words = text.lower().split()
                feat[i, 0] = len(text) / 500.0
                feat[i, 1] = len(words) / 100.0
            depths = self.predict_depth(feat)

        for i, seg in enumerate(segments):
            depth = depths[i] if i < len(depths) else 0
            if hasattr(seg, "depth"):
                seg.depth = depth
        return segments
