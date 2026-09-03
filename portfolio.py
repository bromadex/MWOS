"""
Singles-portfolio builder.

Given a table of value-bet rows (from mwos_edges.compute_market_edges), pick a
subset of N bets — one per fixture — that maximises expected value while
satisfying constraints:
  - each odds >= min_odds        (so any k/N winning combo covers the losses)
  - each model probability >= min_model_prob   (avoids reckless underdog bets)
  - one bet per fixture           (avoids correlated legs from the same match)

Then compute portfolio statistics using the model's per-leg probabilities:
  - Expected value E[P/L]
  - Variance / SD of P/L         (assuming leg independence)
  - Poisson-binomial P(k wins)    (exact via convolution)
  - Break-even hit count
  - Worst-case P/L given exactly k wins (2 highest-odds picks assumed to lose)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_singles_portfolio(
    edges: pd.DataFrame,
    max_bets: int = 10,
    min_odds: float = 1.30,
    min_model_prob: float = 0.60,
    one_per_fixture: bool = True,
) -> pd.DataFrame:
    """Return the selected legs, sorted by kickoff then EV."""
    if edges.empty:
        return edges.copy()
    cand = edges.copy()
    if "both_rated" in cand.columns:
        cand = cand[cand["both_rated"]]
    cand = cand[cand["mwos_odds"].astype(float) >= float(min_odds)]
    cand = cand[cand["p_blend"].astype(float) >= float(min_model_prob)]
    cand = cand[cand["ev_per_unit"].astype(float) > 0]
    if cand.empty:
        return cand

    cand = cand.sort_values("ev_per_unit", ascending=False)
    if one_per_fixture:
        cand = cand.drop_duplicates(subset=["date", "kickoff", "home", "away"], keep="first")
    picked = cand.head(int(max_bets)).copy()
    return picked.sort_values(["date", "kickoff", "home"]).reset_index(drop=True)


def _poisson_binomial_pmf(probs: np.ndarray) -> np.ndarray:
    """Exact P(X=k) for the sum of independent Bernoullis with the given probs.
    O(n^2). n<=10 here so this is trivial."""
    pmf = np.array([1.0])
    for p in probs:
        pmf = np.concatenate(([pmf[0] * (1 - p)], pmf[1:] * (1 - p) + pmf[:-1] * p, [pmf[-1] * p]))
    return pmf


def portfolio_stats(portfolio: pd.DataFrame, stake: float = 1.0) -> dict:
    n = len(portfolio)
    if n == 0:
        return {"n": 0}

    odds = portfolio["mwos_odds"].to_numpy(dtype=float)
    probs = portfolio["p_blend"].to_numpy(dtype=float)

    profit_per_leg_if_win = stake * (odds - 1.0)     # net gain per winning leg
    loss_per_leg_if_lose = -stake                    # net loss per losing leg

    # Per-leg EV and variance (binary outcome)
    leg_ev = probs * profit_per_leg_if_win + (1 - probs) * loss_per_leg_if_lose
    leg_var = probs * (profit_per_leg_if_win - leg_ev) ** 2 + (1 - probs) * (loss_per_leg_if_lose - leg_ev) ** 2

    total_stake = float(stake * n)
    expected_pl = float(leg_ev.sum())
    var_pl = float(leg_var.sum())              # independence assumed
    sd_pl = float(np.sqrt(var_pl))

    # Exact P(exactly k wins) via Poisson-binomial convolution
    pmf = _poisson_binomial_pmf(probs)
    p_at_least = np.array([pmf[k:].sum() for k in range(n + 1)])

    # Worst-case P/L given exactly k winners:
    #   the k winners are the k LOWEST-odds legs
    #   (i.e. the (n-k) HIGHEST-odds legs are the losers)
    odds_sorted_asc = np.sort(odds)  # smallest first
    worst_case_pl_by_k = []
    for k in range(n + 1):
        if k == 0:
            worst_case_pl_by_k.append(-float(total_stake))
            continue
        winners_odds = odds_sorted_asc[:k]  # lowest odds win
        losers_count = n - k
        wc = stake * (winners_odds.sum() - k) - stake * losers_count
        worst_case_pl_by_k.append(float(wc))

    # Break-even hit count under worst-case ordering
    break_even_k = None
    for k in range(n + 1):
        if worst_case_pl_by_k[k] >= 0:
            break_even_k = k
            break

    return {
        "n": n,
        "total_stake": total_stake,
        "expected_pl": expected_pl,
        "sd_pl": sd_pl,
        "expected_winners": float(probs.sum()),
        "sd_winners": float(np.sqrt((probs * (1 - probs)).sum())),
        "pmf_wins": pmf.tolist(),                       # P(exactly k)
        "p_at_least_k_wins": p_at_least.tolist(),       # P(>= k)
        "worst_case_pl_by_k": worst_case_pl_by_k,       # if k win, worst-case net
        "p_no_loss": float(sum(pmf[k] for k in range(n + 1) if worst_case_pl_by_k[k] >= 0)),
        "break_even_k_worst_case": break_even_k,
    }
