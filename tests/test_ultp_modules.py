"""Testes para os módulos ULTP — Wasserstein, Constituency, Coreference, SimCSE."""

import numpy as np
import pytest

from app.engine.wasserstein_cost import WassersteinTopologicalCost
from app.engine.simcse_loss import estimate_anisotropy, info_nce_loss
from app.engine.constituency_guardrail import ConstituencyGuardrail
from app.engine.coreference import CoreferenceTracker
from app.engine.segmenter import segment_bcpd, segment_hybrid


class TestWassersteinCost:
    def test_create_cost(self):
        cost = WassersteinTopologicalCost.create()
        if WassersteinTopologicalCost.is_available():
            assert cost is not None
        else:
            assert cost is None

    def test_is_available(self):
        assert isinstance(WassersteinTopologicalCost.is_available(), bool)

    def test_cost_error_value(self):
        if not WassersteinTopologicalCost.is_available():
            pytest.skip("ruptures/POT not available")
        cost = WassersteinTopologicalCost.create()
        signal = np.random.randn(20, 10).astype(np.float32)
        cost.fit(signal)
        err = cost.error(0, 10)
        assert isinstance(err, float)
        assert err >= 0

    def test_cost_small_window(self):
        if not WassersteinTopologicalCost.is_available():
            pytest.skip("ruptures/POT not available")
        cost = WassersteinTopologicalCost.create()
        signal = np.random.randn(10, 5).astype(np.float32)
        cost.fit(signal)
        err = cost.error(0, 5)
        assert err == 0.0


class TestSimCSE:
    def test_info_nce_numpy(self):
        embs = np.random.randn(10, 32).astype(np.float32)
        loss = info_nce_loss(embs, temperature=0.05)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_info_nce_single(self):
        embs = np.random.randn(1, 32).astype(np.float32)
        loss = info_nce_loss(embs, temperature=0.05)
        assert loss == 0.0

    def test_estimate_anisotropy(self):
        embs = np.random.randn(20, 32).astype(np.float32)
        result = estimate_anisotropy(embs)
        assert "anisotropy_score" in result
        assert "condition_number" in result
        assert "uniformity" in result
        assert 0 <= result["anisotropy_score"] <= 2

    def test_anisotropy_few_samples(self):
        embs = np.random.randn(2, 32).astype(np.float32)
        result = estimate_anisotropy(embs)
        assert result["anisotropy_score"] == 1.0


class TestConstituencyGuardrail:
    def test_guardrail_init(self):
        guardrail = ConstituencyGuardrail()
        assert isinstance(guardrail.is_available, bool)

    def test_is_safe_break_no_model(self):
        guardrail = ConstituencyGuardrail()
        assert guardrail.is_safe_break("Texto simples.", 6) is True

    def test_constituent_boundaries_no_model(self):
        guardrail = ConstituencyGuardrail()
        result = guardrail.find_constituent_boundaries("Texto.", 3)
        assert result == [3]


class TestCoreference:
    def test_find_anaphora(self):
        tracker = CoreferenceTracker()
        segments = [
            "O contrato foi assinado.",
            "Este documento estabelece as cláusulas.",
            "Diante do exposto, a parte requerente solicita.",
        ]
        anaphora = tracker.find_anaphora(segments)
        assert len(anaphora) >= 2

    def test_detect_orphan_risk(self):
        tracker = CoreferenceTracker()
        segments = [
            "O contrato foi assinado entre as partes.",
            "Este documento estabelece as regras do acordo.",
            "As obrigações estão descritas no anexo I.",
        ]
        risk = tracker.detect_orphan_risk(segments)
        assert isinstance(risk, list)

    def test_coreference_density(self):
        tracker = CoreferenceTracker()
        result = tracker.estimate_coreference_density(
            "Este tribunal decidiu. O recorrente apelou. Diante do exposto."
        )
        assert "density" in result
        assert "total_anaphora" in result
        assert result["total_anaphora"] > 0

    def test_resolve_segments(self):
        tracker = CoreferenceTracker()
        segments = [
            "Capítulo I - Do Contrato Social.",
            "Este documento regula as atividades da empresa.",
            "Parágrafo único. O prazo de vigência é de 99 anos.",
        ]
        resolved = tracker.resolve_segments(segments)
        assert len(resolved) >= 1


class TestBCPDIntegration:
    def test_segment_bcpd_basic(self, test_pdf_path):
        from app.engine.extractor import extract_text
        result = extract_text(str(test_pdf_path))
        seg_result = segment_bcpd(result["text"], result.get("blocks"))
        assert isinstance(seg_result.segments, list)
        methods_valid = ("bcpd_no_data", "bcpd_too_short", "bcpd_unavailable",
                         "bcpd_pelt", "bcpd_pelt_wasserstein", "bcpd_pelt_simcse",
                         "bcpd_pelt_wasserstein_simcse")
        if seg_result.method:
            assert seg_result.method in methods_valid, f"Unexpected method: {seg_result.method}"

    def test_segment_hybrid(self, test_pdf_path):
        from app.engine.extractor import extract_text
        result = extract_text(str(test_pdf_path))
        seg_result = segment_hybrid(result["text"], result.get("blocks"))
        if seg_result.segments:
            first = seg_result.segments[0]
            assert hasattr(first, "text")
            assert hasattr(first, "start_char")
            assert hasattr(first, "end_char")
            assert hasattr(first, "depth")

    def test_segment_bcpd_with_embeddings(self, test_pdf_path):
        from app.engine.extractor import extract_text
        result = extract_text(str(test_pdf_path))
        seg_result = segment_bcpd(result["text"], result.get("blocks"), use_embeddings=True)
        assert isinstance(seg_result.segments, list)
