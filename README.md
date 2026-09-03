# MWOS Value-Bet Scanner

A Streamlit web app that reads your daily **MWOS-DAILYFIXTURE PDF** and returns value bets for Spanish football fixtures — La Liga Primera league games plus Copa del Rey / Supercopa cup ties — by comparing a statistical model against MWOS's own price.

**Live:** https://mwos-bet.streamlit.app

---

## What it is

A public, browser-based tool. You drop today's MWOS PDF onto the page; it lists fixtures with positive expected value (EV) across ten markets — 1X2, double chance (1X / 12 / X2), over/under 2.5 goals, and both-teams-to-score yes/no.

It does **not** place bets, does not fund an account, does not connect to any bookmaker. It only *scores* the PDF you upload.

---

## What it really does

1. **Parses** the MWOS PDF into fixtures + 10 markets per match. Recognises any team that has played in La Liga Primera or Segunda in the last decade.
2. **Predicts** each outcome's probability using a Poisson goals model built from openfootball results + Understat xG.
3. **De-vigs** the MWOS 1X2 / OU 2.5 / BTTS lines to get the market's own implied probabilities.
4. **Blends** model and market probabilities (weight adjustable in the sidebar).
5. **Computes EV** for each market vs. the actual MWOS odds; flags positive-EV bets with quarter-Kelly stake suggestions.
6. **Flags cup fixtures** where an opponent is not in the model's rating table as ⚪ **UNRATED** — the raw MWOS odds are shown but no EV is trusted.

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

For home $i$ vs. away $j$ (xG version shown; goals version is identical with $\bar{\lambda}_G$, $H_G$, $\text{hf}_i^G$):

$$\lambda_{H}^{xG} = \bar{\lambda}_{xG} \cdot \alpha_i \cdot \delta_j \cdot H_{xG} \cdot \text{hf}_i^{xG}$$

$$\lambda_{A}^{xG} = \bar{\lambda}_{xG} \cdot \alpha_j \cdot \delta_i$$

Blend the two lambda types:

$$\lambda = 0.70\,\lambda^{xG} + 0.30\,\lambda^{goals}$$

Floored at $0.15$.

**Team-specific home lift.** $H_{xG}$ is the league-wide home boost; $\text{hf}_i^{xG}$ is $i$'s own multiplier on top, computed from its ratio of home xG to away xG divided by the league's ratio, then shrunk toward 1.0 with $k = 20$ matches and clipped to $[0.70,\ 1.40]$. Bilbao's hf $\approx 1.08$, Barcelona's $\approx 0.96$ — Barcelona's away games are strong enough that their marginal home boost is smaller than average.

**Rest & fixture-congestion adjustment.** After the lambda is computed it is trimmed to reflect fatigue:

$$\lambda \leftarrow \lambda - 0.03 \cdot \max(0,\ 4 - \text{rest days}) - 0.10 \cdot \mathbb{1}[\text{played UCL/UEL in last 4 days}]$$

Rest days come from openfootball's La Liga Primera + Segunda schedules. UCL/UEL data is pulled if available; when it isn't, the Euro flag simply stays False and only the rest-days term fires.

**Unrated opponents.** When one team is not in the model's rating table (typical for a lower-division cup side), the app substitutes the neutral fallback $\alpha = \delta = \text{hf} = 1.0$ for the missing side so the equations still resolve. The resulting row is tagged UNRATED and its EV is not shown as actionable.

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
openfootball es.1 (Primera) + es.2 (Segunda) ─┐
openfootball cl / el (Euro fixtures)           ├─→ strengths + rosters + fixture context ──→ snapshot/*.json  (committed)
Understat xG (Primera only)                    │
football-data.co.uk odds (Primera)             ─┘

You upload MWOS PDF
        │
        ▼
parse fixtures + 10 markets (filter to Spanish football via known-teams roster)
        │
        ├── both teams rated ──→ lookup rest / UCL midweek from fixture context
        │                            │
        │                            ▼
        │            Poisson score matrix (λ adjusted for fatigue & team home-lift)
        │                            │
        │                            ▼
        │                        model probs
        │                            │
        │       MWOS line ─── de-vig ─┤
        │                             ▼
        │                    blend (w=0.35)
        │                            │
        │                            ▼
        │              EV vs MWOS odds ──→ flagged bets
        │
        └── unrated opponent → shown separately (UNRATED), no EV computed
```

---

## The website

- **URL:** https://mwos-bet.streamlit.app
- **Hosting:** Streamlit Community Cloud (free tier). Backed by this GitHub repo — every commit to `main` auto-redeploys within ~60 seconds.
- **State:** stateless. No accounts, no history, no data saved between sessions. Uploads live only in the current tab's memory (`st.session_state`).
- **Cold start:** ~5 s.
- **Warm:** near-instant.

### Pages

The app is a **multi-page Streamlit app** (files under `pages/` become separate pages in the left nav):

1. **⚽ Scanner** (`webapp.py`) — the default page. Upload the daily MWOS PDF, get every value bet flagged per fixture.
2. **🎯 Bet Builder** (`pages/1_Bet_Builder.py`) — takes the Scanner's flagged edges and builds a $1-per-leg singles slip. Filters by min odds + min model probability, then computes the full portfolio distribution: expected P/L, standard deviation, Poisson-binomial P(k wins), worst-case P/L per k, and a "safe on k/N hits" verdict. Designed for the classic *"if I win 8 of 10, do I break even?"* question.

### How the website behaves

1. **On load** — reads the committed snapshot (`strengths.json`, `recent_teams.json`, `known_teams.json`, `fixture_context.json`, `meta.json`, ~30 KB total) into memory. Shows four counters at the top: played matches, xG matched, historical odds matched, and **Teams (rated / known)** — the rated count is teams the model can score; the known count is teams the parser will accept from the PDF.
2. **You upload a PDF** — file uploads to the Streamlit server (capped at 20 MB in `.streamlit/config.toml`).
3. **Parse** (~2 s) — `mwos_pdf.py` scans every line, keeps fixtures where both teams are in the known-teams roster (drops other leagues in the PDF), and pulls the 10 odds columns.
4. **Score** (~1 s) — for each fixture, `snapshot_loader.context_for()` looks up each team's rest days, matches-in-14d, and UCL/UEL-midweek flag from the fixture context. λ is adjusted for fatigue and per-team home lift, then the model runs. For cup fixtures with an unrated opponent, league-average fallback ratings are used and the row is marked UNRATED. Model + de-vigged MWOS get blended and EV per market is computed.
5. **Render** — banner if unrated fixtures are present, table of parsed fixtures, summary metrics (Flagged / STRONG / BUY / MARGINAL / Unrated cup), per-fixture expandable cards showing rest days & congestion inline, listing value-bet markets, a dedicated "Unrated cup fixtures" section, and download buttons for the CSVs and text report.

### What you can adjust on the website

Sidebar controls (all live — sliders re-run the score without re-uploading):

| Control | Range | Default | Effect |
|---|---|---|---|
| **Model weight in blend** | 0.00 – 1.00 | 0.35 | Higher = trust model more; lower = anchor to market. Backtest optimum was ≈ 0.00, so lower (0.10–0.25) is safer. |
| **Minimum EV per unit** | 0.00 – 0.20 | 0.02 | Threshold to flag as a bet. Raise to 0.05 to see only serious edges. |
| **Min recent matches per team (2y)** | 5 – 40 | 12 | Filter applied when *computing* the "recent top-flight" set inside `precompute.py`. On the deployed app this control is stored but the effective rated set is baked into the snapshot; change it and re-run `precompute.py` locally to take effect. |

### Signal tiers

| Badge | Meaning | Trigger |
|---|---|---|
| 🟢 **STRONG** | Big EV, both teams rated | EV ≥ 8% and both teams in ratings |
| 🔵 **BUY** | Solid EV, both teams rated | 4% ≤ EV < 8% and both teams in ratings |
| 🟡 **MARGINAL** | Small EV, both teams rated | min-EV ≤ EV < 4% and both teams in ratings |
| ⚪ **UNRATED** | Cup fixture with a team the model can't score | Either team missing from the ratings table |
| ⚪ **SKIP** | Below min-EV threshold | Not shown in the value-bet list |

Never treat an UNRATED row's EV number as real. It's only there so the market for that fixture is visible.

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

Cup / UNRATED fixtures are not in the backtest because there's no equivalent historical market baseline for them.

---

## Repo layout

```
webapp.py              Streamlit entry point — Scanner page (Cloud)
pages/                 Streamlit auto-picks these up as extra pages
  1_Bet_Builder.py       singles-portfolio construction + stats
portfolio.py           slip selection + Poisson-binomial P/L distribution
snapshot_loader.py     loads precomputed team strengths + rosters
snapshot/              committed model state (~30 KB total)
  strengths.json         per-team attack/defense + home-factor multipliers (~36 teams)
  recent_teams.json      teams with enough recent top-flight data           (~23 teams)
  known_teams.json       Primera + Segunda historical roster                (~73 teams)
  fixture_context.json   per-team recent + upcoming matches (60d back / 21d ahead)
                         for rest days, congestion count, and UCL/UEL flag
  meta.json              generation timestamp + counts

config.py              constants (half-life, blend weight, floors)
ratings.py             recency-decayed, shrunk team strengths
model.py               Poisson + Dixon-Coles score matrix
mwos_pdf.py            PDF → fixtures + odds
mwos_edges.py          de-vig + blend + EV (with UNRATED fallback)
backtest.py            walk-forward log-loss vs Pinnacle

data.py                openfootball es.1 + es.2 + Understat + odds joins  (LOCAL only)
odds_history.py        football-data.co.uk fetcher                        (LOCAL only)
precompute.py          rebuilds the snapshot                              (LOCAL only)

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

`precompute.py` needs `requirements-dev.txt` (includes `soccerdata`, which drives a headless Chrome). Streamlit Cloud never runs it — the cloud instance only reads the committed `snapshot/`. Refreshing also updates `known_teams.json`, so newly promoted / relegated clubs are picked up automatically.

---

## Deploying your own copy

1. Fork this repo.
2. https://share.streamlit.io → **New app** → repo = your fork, branch = `main`, main file = `webapp.py`.
3. Deploy. First build ~90 s, then always cached.

---

## Limitations and caveats

- **Ratings are Primera-quality only.** The model has full attack/defense ratings for teams that have played recent Primera football. Cup fixtures (Copa del Rey, Supercopa) featuring lower-division opponents are still parsed and shown, but flagged ⚪ **UNRATED** — the app falls back to league-average multipliers for the missing side so you can see the market but not the EV. Do not stake on UNRATED rows.
- **Rest and UCL/UEL fatigue are modeled** (small −0.03 to −0.12 goals adjustment); injuries, lineups, weather, ref, motivation, and travel distance are **not**. The closing line encodes all of these; the model does not.
- **No live/in-play.** The PDF is a snapshot; odds move.
- **Understat xG only covers Primera.** Segunda teams that later appear as cup opponents have no xG history. Rated Primera clubs playing in cups still get their full ratings from Understat + goals.
- **Understat needs Chrome.** Only runs during local `precompute.py`, never on Cloud.
- **Model log-loss is ~3% behind Pinnacle closing.** Do not bet blindly.
- **Bookmaker limits.** Any bookmaker (MWOS included) will restrict or ban winning accounts. Model edge is not a licence to size aggressively.

**Kelly/4** is the recommended stake. Never full-Kelly. Never chase.
