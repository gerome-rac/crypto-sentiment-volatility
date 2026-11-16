"""Added 7 Oct 2026 (lead): Giacomini-White reports degenerate cases instead of using a pseudo-inverse."""
import numpy as np
from model.tests_eval import giacomini_white


def test_gw_degenerate_when_differences_on_one_day():
    a = np.zeros(500); b = np.zeros(500); b[100] = 1.0
    g = giacomini_white(a, b, conditional=True)
    assert g["degenerate"] and np.isnan(g["p"]) and g["n_nonzero"] == 1


def test_gw_regular_case_matches_solve():
    rng = np.random.default_rng(1)
    a = rng.normal(size=800) ** 2; b = rng.normal(size=800) ** 2
    g = giacomini_white(a, b, conditional=True)
    d = a - b
    Z = np.column_stack([np.ones(len(d) - 1), d[:-1]]) * d[1:, None]
    zbar = Z.mean(0); Om = Z.T @ Z / len(Z)
    assert not g["degenerate"] and np.isclose(g["gw_stat"], len(Z) * zbar @ np.linalg.solve(Om, zbar))


def test_gw_identical_forecasts():
    a = np.ones(300)
    g = giacomini_white(a, a.copy())
    assert g["degenerate"] and g["p"] == 1.0
