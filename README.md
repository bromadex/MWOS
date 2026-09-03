# MWOS Value-Bet Scanner

A Streamlit web app that reads your daily **MWOS-DAILYFIXTURE PDF** and returns value bets for La Liga fixtures by comparing a statistical model against MWOS's own price.

**Live:** https://mwos-bet.streamlit.app

---

## What it is

A public, browser-based tool. You drop today's MWOS PDF onto the page; it lists fixtures with positive expected value (EV) across ten markets — 1X2, double chance (1X / 12 / X2), over/under 2.5 goals, and both-teams-to-score yes/no.

It does **not** place bets, does not fund an account, does not connect to any bookmaker. It only *scores* the PDF you upload.

---

## What it really does

1. **Parses** the MWOS PDF into fixtures + 10 markets per match.
2. **Predicts** each outcome's probability using a Poisson goals model built from openfootball results + Understat xG.
3. **De-vigs** the MWOS 1X2 / OU 2.5 / BTTS lines to get the market's own implied probabilities.
4. **Blends** model and market probabilities (weight adjustable in the sidebar).
5. **Computes EV** for each market vs. the actual MWOS odds; flags positive-EV bets with quarter-Kelly stake suggestions.

---

## Distribution

Goals are modelled as **independent Poisson** random variables (one for home, one for away), corrected by the **Dixon-Coles** adjustment for known mispricing at low scores (0-0, 1-0, 0-1, 1-1).

---

## The math

### 1. Recency-weighted match weights

Half-life $H = 540$ days:

$$w_m = \exp\left(-\ln 2 \cdot \frac{\text{AgeDays}_m}{H}\right)$$

### 2. Team strengths (xG-based, goals-based analogously)

Attack rate:

$$A_i = \frac{\sum_m w_m \cdot \text{xG For}_{i,m}}{\sum_m w_m}$$

Defensive weakness:

$$D_i = \frac{\sum_m w_m \cdot \text{xG Against}_{i,m}}{\sum_m w_m}$$

### 3. Shrinkage toward the league mean

Every rating is blended toward the league mean as if the team had $k = 8$ extra average matches, so promoted teams and small samples don't produce extreme ratings:

$$\hat{A}_i = \frac{\sum_m w_m x_m \;+\; k \cdot \bar{A}}{\sum_m w_m \;+\; k}$$

Then normalised to multipliers:

$$\alpha_i = \hat{A}_i / \bar{A}, \qquad \delta_i = \hat{D}_i / \bar{D}$$

Clipped to $[0.50,\ 1.80]$.

### 4. Fixture expected goals

For home $i$ vs. away $j$ (xG version shown; goals version is identical with $\bar{\lambda}_G$ and $H_G$):

$$\lambda_{H}^{xG} = \bar{\lambda}_{xG} \cdot \alpha_i \cdot \delta_j \cdot H_{xG}$$

$$\lambda_{A}^{xG} = \bar{\lambda}_{xG} \cdot \alpha_j \cdot \delta_i$$

Blend the two lambda types:

$$\lambda = 0.70\,\lambda^{xG} + 0.30\,\lambda^{goals}$$

Floored at $0.15$.

### 5. Score matrix

$$P(H=k) = \frac{\lambda_H^k \, e^{-\lambda_H}}{k!}, \qquad P(H=k, A=l) = P(H=k)\,P(A=l)$$

### 6. Dixon-Coles low-score correction

$$
\tau_{k,l} =
\begin{cases}
1 - \lambda_H \lambda_A \rho & (k,l)=(0,0) \\
1 + \lambda_H \rho & (k,l)=(0,1) \\
1 + \lambda_A \rho & (k,l)=(1,0) \\
1 - \rho & (k,l)=(1,1) \\
1 & \text{otherwise}
\end{cases}
$$

$\rho$ is refit inside each backtest fold to avoid leakage.

### 7. Market probabilities from the score matrix

$$P(\text{Home}) = \sum_{k>l} P(k,l), \quad P(\text{Draw}) = \sum_{k=l} P(k,l), \quad P(\text{Away}) = \sum_{k<l} P(k,l)$$

$$P(\text{Over 2.5}) = \sum_{k+l\ge 3} P(k,l), \qquad P(\text{BTTS}) = \sum_{k\ge 1,\, l\ge 1} P(k,l)$$

### 8. De-vig the MWOS line

For MWOS decimal odds $O_H, O_D, O_A$:

$$p_H^{\text{mkt}} = \frac{1/O_H}{1/O_H + 1/O_D + 1/O_A}$$

MWOS overround per market = $\sum 1/O - 1$ (typical 4%–11%). Same recipe for OU 2.5 and BTTS 2-way.

### 9. Blend

$$p_{\text{final}} = w \cdot p_{\text{model}} + (1 - w) \cdot p_{\text{mkt}}$$

Default $w = 0.35$. Adjustable in the sidebar.

### 10. Expected value and Kelly

$$\text{EV per unit staked} = p_{\text{final}} \cdot O_{\text{MWOS}} - 1$$

Full-Kelly optimal fraction:

$$f^{\star} = \frac{p_{\text{final}} \cdot O_{\text{MWOS}} - 1}{O_{\text{MWOS}} - 1}$$

Recommended stake = $f^{\star}/4$ (quarter-Kelly). The app shows this as **Kelly/4**.

---

## Pipeline

```
openfootball JSON  ─┐
Understat xG        ├─→ team strengths ──→ snapshot/*.json  (committed)
football-data odds  ─┘

You upload MWOS PDF
        │
        ▼
parse fixtures + 10 markets ──→ Poisson score matrix ──→ model probs
                                                             │
MWOS line ─── de-vig ──→ market probs ───────────────────────┤
                                                             ▼
                                                blend (w=0.35)
                                                             │
                                                             ▼
                                          EV vs MWOS odds ──→ flagged bets
```

---

## The website

- **URL:** https://mwos-bet.streamlit.app
- **Hosting:** Streamlit Community Cloud (free tier). Backed by this GitHub repo — every commit to `main` auto-redeploys within ~60 seconds.
- **State:** stateless. No accounts, no history, no data saved between sessions. Uploads live only in the current tab's memory.
- **Cold start:** ~5 s (Streamlit spins the process up on first visit after idle).
- **Warm:** near-instant.

### How the website behaves

1. **On load** — reads `snapshot/strengths.json` (7 KB, committed) into memory. Shows four counters at the top: played matches, xG matched, historical odds matched, teams covered, plus the snapshot's generation timestamp.
2. **You upload a PDF** — file uploads to the Streamlit server (capped at 20 MB in `.streamlit/config.toml`).
3. **Parse** (~2 s) — `mwos_pdf.py` scans every line, extracts fixtures whose teams match the recent-top-flight set, and pulls the 10 odds columns.
4. **Score** (~1 s) — model runs, blends with the de-vigged MWOS line, computes EV and Kelly/4 for every market.
5. **Render** — table of parsed fixtures, a per-fixture expandable card listing every value-bet market, summary counters (STRONG / BUY / MARGINAL), and download buttons for the CSVs and text report.

### What you can adjust on the website

Sidebar controls (all live — sliders re-run the score without re-uploading):

| Control | Range | Default | Effect |
|---|---|---|---|
| **Model weight in blend** | 0.00 – 1.00 | 0.35 | Higher = trust model more; lower = anchor to market. Backtest optimum was ≈ 0.00, so lower (0.10–0.25) is safer. |
| **Minimum EV per unit** | 0.00 – 0.20 | 0.02 | Threshold to flag as a bet. Raise to 0.05 to see only serious edges. |
| **Min recent matches per team (2y)** | 5 – 40 | 12 | Fixtures where either team has fewer matches than this in the last two years are skipped (filters out stale ratings for recently promoted/relegated teams). |

Signals rendered per market:
- 🟢 **STRONG** — EV ≥ 8%
- 🔵 **BUY** — EV ≥ 4%
- 🟡 **MARGINAL** — EV ≥ min-EV slider

---

## Backtest evidence

Walk-forward, 5 folds over 5,310 La Liga matches (2,806 with market data):

| Predictor | Log-loss |
|---|---|
| Uniform (1/3 each) | 1.099 |
| League H/D/A average | 1.063 |
| **Model alone** | 0.998 |
| Blend w = 0.35 | 0.979 |
| Blend w = 0.10 | 0.972 |
| **Pinnacle closing (de-vigged)** | **0.971** |

The model beats the naïve baseline by ~6% log-loss but is ~3% behind Pinnacle. Since MWOS's overround is 4%–11%, the model is not a guaranteed money-maker against MWOS. Use it as a screening / disagreement flag, not a black box. Backtest is reproducible via `python main.py backtest`.

---

## Repo layout

```
webapp.py              Streamlit entry point (Cloud)
snapshot_loader.py     loads precomputed team strengths
snapshot/              committed model state (~8 KB total)
  strengths.json         per-team attack/defense multipliers
  recent_teams.json      teams with enough recent top-flight data
  meta.json              generation timestamp + counts

config.py              constants (half-life, blend weight, floors)
ratings.py             recency-decayed, shrunk team strengths
model.py               Poisson + Dixon-Coles score matrix
mwos_pdf.py            PDF → fixtures + odds
mwos_edges.py          de-vig + blend + EV
backtest.py            walk-forward log-loss vs Pinnacle

data.py                openfootball + Understat + odds joins  (LOCAL only)
odds_history.py        football-data.co.uk fetcher            (LOCAL only)
precompute.py          rebuilds the snapshot                  (LOCAL only)

main.py                CLI (predict, backtest, mwos, daily)
run_web.bat            local Streamlit launcher
requirements.txt       Cloud dependencies (no headless Chrome)
requirements-dev.txt   local extras (adds soccerdata)
.streamlit/config.toml theme + upload cap
```

---

## Refreshing the model

Ratings drift as matches are played. Recommended cadence: **weekly during the season**.

```bash
python precompute.py         # pulls fresh data, rebuilds snapshot/
git add snapshot/
git commit -m "refresh snapshot"
git push                     # Streamlit Cloud redeploys automatically
```

`precompute.py` needs `requirements-dev.txt` (includes `soccerdata`, which drives a headless Chrome). Streamlit Cloud never runs it — the cloud instance only reads the committed `snapshot/`.

---

## Deploying your own copy

1. Fork this repo.
2. https://share.streamlit.io → **New app** → repo = your fork, branch = `main`, main file = `webapp.py`.
3. Deploy. First build ~90 s, then always cached.

---

## Limitations and caveats

- **La Liga Primera only.** Fixtures involving lower-division teams are silently skipped.
- **No injuries, lineups, weather, ref, motivation, or travel.** The closing line encodes all of these; the model does not.
- **No live/in-play.** The PDF is a snapshot; odds move.
- **Understat xG needs Chrome.** Only runs during local `precompute.py`, never on Cloud.
- **Model log-loss is ~3% behind Pinnacle closing.** Do not bet blindly.
- **Bookmaker limits.** Any bookmaker (MWOS included) will restrict or ban winning accounts. Model edge is not a licence to size aggressively.

**Kelly/4** is the recommended stake. Never full-Kelly. Never chase.
