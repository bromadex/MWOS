from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from config import (
    MIN_EXPECTED_GOALS,
    SCORE_MATRIX_MAX_GOALS,
    XG_LAMBDA_WEIGHT,
)


def fixture_lambdas(home: str, away: str, strengths: dict) -> tuple[float, float]:
    teams = strengths["teams"]
    if home not in teams or away not in teams:
        raise KeyError(f"missing team in strengths: {home} / {away}")

    h = teams[home]
    a = teams[away]

    lam_h_xg = strengths["league_mean_xg"] * h["att_xg"] * a["def_xg"] * strengths["home_adv_xg"]
    lam_a_xg = strengths["league_mean_xg"] * a["att_xg"] * h["def_xg"]
    lam_h_g = strengths["league_mean_goals"] * h["att_g"] * a["def_g"] * strengths["home_adv_goals"]
    lam_a_g = strengths["league_mean_goals"] * a["att_g"] * h["def_g"]

    lam_h = XG_LAMBDA_WEIGHT * lam_h_xg + (1 - XG_LAMBDA_WEIGHT) * lam_h_g
    lam_a = XG_LAMBDA_WEIGHT * lam_a_xg + (1 - XG_LAMBDA_WEIGHT) * lam_a_g

    return max(lam_h, MIN_EXPECTED_GOALS), max(lam_a, MIN_EXPECTED_GOALS)


def _dixon_coles_tau(k: int, l: int, lam_h: float, lam_a: float, rho: float) -> float:
    if k == 0 and l == 0:
        return 1 - lam_h * lam_a * rho
    if k == 0 and l == 1:
        return 1 + lam_h * rho
    if k == 1 and l == 0:
        return 1 + lam_a * rho
    if k == 1 and l == 1:
        return 1 - rho
    return 1.0


def score_matrix(lam_h: float, lam_a: float, rho: float = 0.008, max_g: int = SCORE_MATRIX_MAX_GOALS) -> np.ndarray:
    ks = np.arange(max_g + 1)
    p_h = poisson.pmf(ks, lam_h)
    p_a = poisson.pmf(ks, lam_a)
    m = np.outer(p_h, p_a)

    for k in range(2):
        for l in range(2):
            m[k, l] *= _dixon_coles_tau(k, l, lam_h, lam_a, rho)

    total = m.sum()
    if total > 0:
        m = m / total
    return m


def market_probabilities(m: np.ndarray) -> dict:
    n = m.shape[0]
    idx_h = np.arange(n)[:, None]
    idx_a = np.arange(n)[None, :]

    p_home = float(m[idx_h > idx_a].sum())
    p_draw = float(np.trace(m))
    p_away = float(m[idx_h < idx_a].sum())

    over25 = 0.0
    btts = 0.0
    for k in range(n):
        for l in range(n):
            if k + l >= 3:
                over25 += m[k, l]
            if k >= 1 and l >= 1:
                btts += m[k, l]

    most_likely = np.unravel_index(np.argmax(m), m.shape)

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "over_2_5": float(over25),
        "btts": float(btts),
        "most_likely_score": f"{most_likely[0]}-{most_likely[1]}",
    }


def predict_fixture(home: str, away: str, strengths: dict, rho: float = 0.008) -> dict:
    lam_h, lam_a = fixture_lambdas(home, away, strengths)
    m = score_matrix(lam_h, lam_a, rho)
    mp = market_probabilities(m)
    return {
        "home": home,
        "away": away,
        "lambda_h": round(lam_h, 3),
        "lambda_a": round(lam_a, 3),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in mp.items()},
    }
