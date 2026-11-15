"""Daily-high PDF = the kmia2h climate-day high, floored at the proven recorded high (no horizon weighting)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalshi_api import algorithm as alg  # noqa: E402


def _entry(temps, probs):
    return {"temps": temps, "probs": probs, "high_temps": temps, "high_probs": probs}


def test_day_high_pdf_comes_only_from_the_climate_day_entry():
    results = {"kmia2h_day_high": _entry([88, 89], [0.25, 0.75]),
               "kmia2h_high_2h": _entry([80], [1.0]), "kmia2h_low_2h": _entry([70], [1.0])}
    assert alg._get_day_high_pdf(results) == {88: pytest.approx(0.25), 89: pytest.approx(0.75)}


def test_day_high_pdf_is_normalised():
    pdf = alg._get_day_high_pdf({"kmia2h_day_high": _entry([88, 89], [2.0, 2.0])})
    assert pdf == {88: pytest.approx(0.5), 89: pytest.approx(0.5)}


def test_no_pdf_when_the_day_entry_is_missing_or_failed():
    assert alg._get_day_high_pdf({}) == {}
    assert alg._get_day_high_pdf({"kmia2h_day_high": {"error": "ABSTAIN: no recent KMIA observation"}}) == {}
    assert alg._get_day_high_pdf({"kmia2h_high_2h": _entry([88], [1.0])}) == {}     # 2-h entries are not used
    assert alg._compute_daily_high_pdf(85.0, {}) == {}


def test_high_cannot_be_below_the_observed_high():
    pdf = alg._compute_daily_high_pdf(90.2, {86: 0.2, 90: 0.5, 92: 0.3})
    assert min(pdf) == 90                       # 86 mass folded onto the rounded observed high
    assert pdf[90] == pytest.approx(0.7)
    assert pdf[92] == pytest.approx(0.3)
    assert sum(pdf.values()) == pytest.approx(1.0)


def test_no_horizon_weighting_is_left():
    for name in ("HORIZON_WEIGHTS", "HORIZON_MIX", "_get_horizon_pdfs", "_get_path_high_pdf"):
        assert not hasattr(alg, name)
