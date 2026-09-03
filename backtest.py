from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from model import _dixon_coles_tau, fixture_lambdas
from ratings import compute_team_strengths


def naive_baselines(played: pd.DataFrame) -> dict:
    """Log-loss for trivial baselines so the model's number has context."""
    df = played.dropna(subset=["home_goals", "away_goals"]).copy()
    outcomes = np.where(
        df["home_goals"] > df["away_goals"], "H",
        np.where(df["home_goals"] < df["away_goals"], "A", "D"),
    )
    n = len(outcomes)
    p_h_emp = float((outcomes == "H").mean())
    p_d_emp = float((outcomes == "D").mean())
    p_a_emp = float((outcomes == "A").mean())

    def _ll(p_map):
        return float(-np.mean([np.log(max(p_map[o], 1e-9)) for o in outcomes]))

    return {
        "uniform_1x2": _ll({"H": 1 / 3, "D": 1 / 3, "A": 1 / 3}),
        "always_home": _ll({"H": 1.0 - 2e-9, "D": 1e-9, "A": 1e-9}),
        "league_avg_1x2": _ll({"H": p_h_emp, "D": p_d_emp, "A": p_a_emp}),
        "sample_hda_share": (round(p_h_emp, 3), round(p_d_emp, 3), round(p_a_emp, 3)),
        "n": n,
    }


def _match_probs(lam_h: float, lam_a: float, rho: float, max_g: int = 10):
    ks = np.arange(max_g + 1)
    p_h = poisson.pmf(ks, lam_h)
    p_a = poisson.pmf(ks, lam_a)
    m = np.outer(p_h, p_a)
    for k in range(2):
        for l in range(2):
            m[k, l] *= _dixon_coles_tau(k, l, lam_h, lam_a, rho)
    m = m / m.sum()
    ph = m[np.arange(max_g + 1)[:, None] > np.arange(max_g + 1)[None, :]].sum()
    pd_ = np.trace(m)
    pa = m[np.arange(max_g + 1)[:, None] < np.arange(max_g + 1)[None, :]].sum()
    return ph, pd_, pa


def _fit_rho(train: pd.DataFrame, strengths: dict) -> float:
    rows = []
    for _, r in train.iterrows():
        try:
            lh, la = fixture_lambdas(r["home"], r["away"], strengths)
            rows.append((lh, la, int(r["home_goals"]), int(r["away_goals"])))
        except Exception:
            continue
    if not rows:
        return 0.0

    def nll(rho):
        s = 0.0
        for lh, la, hg, ag in rows:
            p = poisson.pmf(hg, lh) * poisson.pmf(ag, la) * _dixon_coles_tau(min(hg, 1), min(ag, 1), lh, la, rho)
            s += -np.log(max(p, 1e-12))
        return s

    res = minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded", options={"xatol": 1e-4})
    return float(res.x)


def walk_forward(
    played: pd.DataFrame,
    half_lives=(540,),
    blend_weights=(1.0, 0.75, 0.5, 0.25, 0.0),
    n_folds: int = 5,
    min_train: int = 2000,
):
    """
    Walk-forward backtest. Reports log-loss for the model, for the market
    (de-vigged Pinnacle closing) and for every blend w*model + (1-w)*market,
    computed on the same held-out fold rows so the numbers are apples-to-apples.
    """
    df = played.dropna(subset=["home_goals", "away_goals", "date"]).sort_values("date").reset_index(drop=True)
    if len(df) < min_train + 200:
        raise RuntimeError(f"not enough data: {len(df)}")

    fold_size = (len(df) - min_train) // n_folds
    results = []

    for hl in half_lives:
        rhos = []
        rows = []
        for fold in range(n_folds):
            train_end = min_train + fold * fold_size
            test_end = train_end + fold_size
            train = df.iloc[:train_end]
            test = df.iloc[train_end:test_end]

            ref = pd.Timestamp(train["date"].max())
            strengths = compute_team_strengths(train, ref_date=ref, half_life=hl)
            rho = _fit_rho(train.tail(1500), strengths)
            rhos.append(rho)

            for _, r in test.iterrows():
                try:
                    lh, la = fixture_lambdas(r["home"], r["away"], strengths)
                except KeyError:
                    continue
                ph, pd_, pa = _match_probs(lh, la, rho)
                hg, ag = int(r["home_goals"]), int(r["away_goals"])
                if hg > ag:
                    outcome = "H"
                elif hg < ag:
                    outcome = "A"
                else:
                    outcome = "D"

                mkt_h = r.get("mkt_p_h")
                mkt_d = r.get("mkt_p_d")
                mkt_a = r.get("mkt_p_a")
                has_mkt = all(pd.notna(x) for x in [mkt_h, mkt_d, mkt_a])

                rows.append(
                    {
                        "outcome": outcome,
                        "p_model_h": ph,
                        "p_model_d": pd_,
                        "p_model_a": pa,
                        "p_mkt_h": float(mkt_h) if has_mkt else None,
                        "p_mkt_d": float(mkt_d) if has_mkt else None,
                        "p_mkt_a": float(mkt_a) if has_mkt else None,
                    }
                )

        eval_df = pd.DataFrame(rows)
        n_total = len(eval_df)
        has_mkt = eval_df["p_mkt_h"].notna()
        overlap = eval_df[has_mkt].reset_index(drop=True)
        n_overlap = len(overlap)

        def _ll(pH, pD, pA, outcomes):
            probs = np.select(
                [outcomes == "H", outcomes == "D", outcomes == "A"],
                [pH, pD, pA],
                default=np.nan,
            )
            probs = np.clip(probs, 1e-9, 1.0)
            return float(-np.log(probs).mean())

        model_ll_full = _ll(eval_df["p_model_h"], eval_df["p_model_d"], eval_df["p_model_a"], eval_df["outcome"].values)

        model_ll = _ll(overlap["p_model_h"], overlap["p_model_d"], overlap["p_model_a"], overlap["outcome"].values) if n_overlap else None
        mkt_ll = _ll(overlap["p_mkt_h"], overlap["p_mkt_d"], overlap["p_mkt_a"], overlap["outcome"].values) if n_overlap else None

        base = {
            "half_life": hl,
            "n_total": n_total,
            "n_with_market": n_overlap,
            "mean_rho": float(np.mean(rhos)),
            "model_ll_all": round(model_ll_full, 4),
            "model_ll_overlap": round(model_ll, 4) if model_ll is not None else None,
            "market_ll_overlap": round(mkt_ll, 4) if mkt_ll is not None else None,
        }

        for w in blend_weights:
            if n_overlap == 0:
                base[f"blend_ll_w{w:.2f}"] = None
                continue
            bH = w * overlap["p_model_h"] + (1 - w) * overlap["p_mkt_h"]
            bD = w * overlap["p_model_d"] + (1 - w) * overlap["p_mkt_d"]
            bA = w * overlap["p_model_a"] + (1 - w) * overlap["p_mkt_a"]
            base[f"blend_ll_w{w:.2f}"] = round(_ll(bH, bD, bA, overlap["outcome"].values), 4)

        results.append(base)
    return pd.DataFrame(results)
