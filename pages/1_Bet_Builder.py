"""Bet Builder — turn today's flagged edges into a singles portfolio.

Statistical framing:
- N independent Bernoulli legs, each with model probability p_i and MWOS
  decimal odds d_i
- Stake S per leg, total exposure = N·S
- Per-leg P/L: +S·(d_i - 1) if win, -S if lose
- Portfolio P/L is a sum of independent RVs
- Number of winning legs follows a Poisson-binomial distribution (unequal p_i)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from portfolio import build_singles_portfolio, portfolio_stats

st.set_page_config(page_title="Bet Builder — MWOS", page_icon="🎯", layout="wide")

st.title("🎯 Bet Builder — $1 singles portfolio")
st.caption(
    "Takes today's flagged edges from the Scanner and picks a subset of legs "
    "you can bet as singles. The stats panel tells you whether the slip is "
    "robust — i.e. whether it still breaks even when only k of N legs win."
)

edges: pd.DataFrame | None = st.session_state.get("edges")
if edges is None or edges.empty:
    st.warning(
        "No edges in memory yet. Open the **Scanner** page (left nav), upload "
        "today's MWOS PDF, then come back here."
    )
    st.stop()

stamp = st.session_state.get("upload_stamp", "")
name = st.session_state.get("upload_name", "current PDF")
st.caption(f"Using edges from **{name}** (loaded {stamp}).")

with st.sidebar:
    st.header("Portfolio controls")
    stake = st.number_input("Stake per leg (units)", min_value=0.1, max_value=1000.0, value=1.0, step=0.5)
    max_bets = st.slider("Max legs on the slip", min_value=3, max_value=15, value=10)
    min_odds = st.slider(
        "Min odds per leg",
        min_value=1.05,
        max_value=2.50,
        value=1.30,
        step=0.05,
        help="Higher min odds = fewer picks but more cushion per winner. "
             "At min_odds=1.25, any 8/10 winning combo guarantees break-even.",
    )
    min_model_prob = st.slider(
        "Min model probability per leg",
        min_value=0.50,
        max_value=0.90,
        value=0.60,
        step=0.05,
        help="Filters out speculative underdog bets. A leg with model prob 0.75 "
             "and odds 1.60 is the sweet spot for the singles strategy.",
    )
    one_per_fixture = st.checkbox(
        "One leg per fixture (avoids correlated bets)",
        value=True,
        help="Prevents e.g. picking 'Barça win' AND 'Barça over 2.5' in the same match.",
    )

port = build_singles_portfolio(
    edges,
    max_bets=int(max_bets),
    min_odds=float(min_odds),
    min_model_prob=float(min_model_prob),
    one_per_fixture=bool(one_per_fixture),
)

if port.empty:
    st.warning(
        "No legs cleared the filters. Try lowering **Min odds** or "
        "**Min model probability**, or upload today's PDF on the Scanner first."
    )
    st.stop()

stats = portfolio_stats(port, stake=float(stake))
n = stats["n"]

# ── the slip ─────────────────────────────────────────────────────────────
st.subheader(f"Selected slip — {n} legs")

view = port[["date", "kickoff", "home", "away", "market", "mwos_odds", "p_blend", "p_model", "ev_per_unit"]].copy()
view["mwos_odds"] = view["mwos_odds"].map(lambda v: f"{v:.2f}")
view["p_blend"] = view["p_blend"].map(lambda v: f"{v*100:.1f}%")
view["p_model"] = view["p_model"].map(lambda v: f"{v*100:.1f}%")
view["ev_per_unit"] = view["ev_per_unit"].map(lambda v: f"{v*100:+.2f}%")
view.columns = ["Date", "Kickoff", "Home", "Away", "Market", "Odds", "p_blend", "p_model", "EV/unit"]
st.dataframe(view, use_container_width=True, hide_index=True)

# ── portfolio summary ────────────────────────────────────────────────────
st.divider()
st.subheader("Portfolio statistics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total stake", f"${stats['total_stake']:.2f}")
c2.metric("Expected P/L", f"${stats['expected_pl']:+.2f}", help="Expected profit assuming model probabilities are correct.")
c3.metric("SD of P/L", f"${stats['sd_pl']:.2f}", help="Standard deviation of the slip's P/L, assuming independent legs.")
c4.metric(
    "P(no net loss)",
    f"{stats['p_no_loss']*100:.1f}%",
    help="Probability the slip nets ≥ $0. Computed via exact Poisson-binomial over the per-leg model probabilities, worst-case ordering of which legs lose.",
)

c5, c6, c7 = st.columns(3)
c5.metric(
    "Expected # of winners",
    f"{stats['expected_winners']:.2f} / {n}",
    help="Sum of per-leg model probabilities.",
)
c6.metric(
    "SD of # winners",
    f"{stats['sd_winners']:.2f}",
)
be = stats["break_even_k_worst_case"]
c7.metric(
    "Break-even hit count (worst case)",
    "impossible" if be is None else f"{be} / {n}",
    help="Minimum number of winning legs needed to net ≥ $0 even if the losing legs happen to be your highest-odds picks.",
)

# ── the "if k win" table ─────────────────────────────────────────────────
st.divider()
st.subheader("Scenario analysis: if exactly k legs win")

pmf = stats["pmf_wins"]
wc_pl = stats["worst_case_pl_by_k"]

rows = []
for k in range(n + 1):
    rows.append(
        {
            "k winners": k,
            "P(exactly k)": f"{pmf[k]*100:.2f}%",
            "Worst-case P/L": f"${wc_pl[k]:+.2f}",
            "Net vs. stake": "✅ break-even or profit" if wc_pl[k] >= 0 else "❌ loss",
        }
    )
scenario = pd.DataFrame(rows)
st.dataframe(scenario, use_container_width=True, hide_index=True)
st.caption(
    "**P(exactly k)** uses the exact Poisson-binomial distribution over the per-leg model probabilities. "
    "**Worst-case P/L** assumes the (N−k) losing legs are your highest-odds picks (the least likely to win) — "
    "the pessimistic side of the range. Actual P/L is at least this, on average higher."
)

# ── verdict banner ───────────────────────────────────────────────────────
st.divider()
target_k = min(8, n)
target_pl = wc_pl[target_k] if target_k < len(wc_pl) else wc_pl[-1]
p_at_least_target = stats["p_at_least_k_wins"][target_k]

if target_pl >= 0:
    st.success(
        f"✅ **Slip is safe on {target_k}/{n} hits** — worst-case P/L "
        f"if exactly {target_k} legs win is **${target_pl:+.2f}**. "
        f"Probability of hitting ≥ {target_k} legs (per the model): **{p_at_least_target*100:.1f}%**."
    )
else:
    st.error(
        f"❌ **Slip loses on {target_k}/{n} hits** (worst-case P/L = ${target_pl:+.2f}). "
        f"Raise **Min odds** or drop the weakest legs. "
        f"Break-even worst-case needs {be}/{n} wins."
    )

# ── math cheatsheet ──────────────────────────────────────────────────────
with st.expander("📐 The math (for the stats project)"):
    st.markdown(
        r"""
**Setup.** $N$ independent Bernoulli legs indexed $i = 1, \ldots, N$, each with:

- model win probability $p_i$
- MWOS decimal odds $d_i$
- stake $S$ per leg

**Per-leg P/L** (random variable):

$$X_i = \begin{cases} +S(d_i - 1) & \text{with prob. } p_i \\ -S & \text{with prob. } 1 - p_i \end{cases}$$

$$\mathbb{E}[X_i] = p_i\,S(d_i - 1) - (1 - p_i)\,S = S(p_i d_i - 1)$$

$$\operatorname{Var}(X_i) = p_i(1 - p_i)\,\big[S(d_i - 1) - (-S)\big]^2 = p_i(1 - p_i)\,S^2 d_i^2$$

**Portfolio P/L** $= \sum X_i$. Under independence:

$$\mathbb{E}[\text{P/L}] = S \sum_{i=1}^{N} (p_i d_i - 1), \qquad \operatorname{SD}(\text{P/L}) = \sqrt{\sum_{i=1}^{N} \operatorname{Var}(X_i)}$$

**Number of winning legs** $W = \sum_{i} B_i$ where $B_i \sim \text{Bernoulli}(p_i)$.
Because the $p_i$ differ, $W$ follows a **Poisson-binomial** distribution
(not binomial). The exact PMF is computed by $O(N^2)$ convolution:

$$P(W = k) = \text{coefficient of } z^k \text{ in } \prod_{i=1}^{N} \big((1 - p_i) + p_i z\big)$$

**Worst-case P/L given $W = k$** (used to compute the "safe on $k/N$" verdict):
assume the $(N - k)$ losers are the legs with the **highest odds**. Then

$$\text{P/L}_{\text{worst}}(k) = S \Big(\sum_{i \in \text{lowest } k \text{ odds}} (d_i - 1)\Big) - S(N - k)$$

**"Safe on $k/N$"** condition: $\text{P/L}_{\text{worst}}(k) \ge 0$, i.e.

$$\sum_{i \in \text{lowest } k \text{ odds}} d_i \ge N$$

For the classic "any $8/10$ hits break even" case, this reduces to
requiring **avg odds of the 8 lowest legs $\ge 1.25$**. In practice, setting
`Min odds = 1.30` makes every leg contribute ≥ $0.30 profit on a win, so any
$k$ winners with $k \ge \lceil N/1.30 \rceil$ break even.
""".strip()
    )
