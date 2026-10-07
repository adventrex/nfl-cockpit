# NFL Edge Cockpit (v4.2)

Pre-game NFL moneyline decision support: market-as-prior win probabilities, EPA-based
model nudge, half-Kelly sizing, same-game-parlay builder, a **hedge desk**, and a
bet log that auto-grades results and closing-line value (CLV) from nflverse.

## Run

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your the-odds-api.com key
.venv/bin/streamlit run app.py
```

Without an odds key the app falls back to the schedule's cached lines.

## Test

```bash
.venv/bin/python -m pytest tests -q
```

## Layout

| file | what |
|---|---|
| `engine.py` | odds math, excess-HFA, win-prob blend, decision + sizing, prop probs, bet-log grading |
| `hedge_engine.py` | lock / Kelly-optimal hedge, spread middles, parlay last-leg hedge |
| `parlay_engine.py` | bounded correlation joint prob, EV vs book price |
| `app.py` | Streamlit UI only |
| `tests/` | 30 pytest cases pinning the math |

## v4 changes vs v3.2 (Dec 2025)

Bugs fixed:
- **"News Variance" was seeded random noise** (±3.2% win prob by default). Alone it could
  trigger a bet at -110. Replaced with a signed manual slider defaulting to 0.
- **Slider double-count**: defaults were derived from team EPA, then re-added on top of the
  same EPA in the model features (~1.75× weight). Sliders now contribute only their
  *delta* from the data-driven default.
- **HFA double-count**: the team's full home-field edge was added on top of a market price
  that already includes it. Now only the excess over league average is added.
- **Weather** penalised the home side; it now shrinks the favorite's edge toward 50/50.
- **Parlay EV was always -15%** (priced from its own probability, then vigged). Now EV is
  computed against the book's parlay price; joint prob is bounded by the smallest leg.
- **Props were hard-coded to 57% at -110** (+9% EV every leg). Now line = recent median,
  P(over) = shrunk hit rate.
- **Weekly player stats 404**: nflverse moved the file; loader uses the new path.
- Removed the fake "last year's schedule shifted +1 year" fallback.
- Removed the hard-coded Odds API key (public repo). Reads `st.secrets` / env var.

Added: multi-book best price + synthetic hold, min-EV buffer, model holdout log-loss vs
market, Hedge Desk (lock / Kelly-optimal / middle finder / parlay last leg), Bet Log with
auto-grading and CLV.

## v4.1 (2026-09-17): injuries, pre-game features, backtest

- `injuries.py` + `scan_injuries.py`: Sleeper injury feed (free, no key) auto-sets the
  News slider from depth-chart starters (QB1 out ≈ −6%, others small, capped). A launchd
  job (`com.andrew.nfl-injury-scan`, 07:00 + 19:00) diffs against the last snapshot and
  iMessages the starter-level changes.
- `features.py`: leakage-free pre-game features. Team EPA is an exponentially-weighted
  mean of *prior* games (half-life 6, 60% offseason carry); QB EPA follows the **player**
  across teams; backup-QB flag; rest diff, divisional, outdoor, wind; official-report
  injury load.
- `backtest.py` → `BACKTEST.md`: walk-forward over 2024–2026 (560 games), five nested
  feature sets, scored at the closing moneyline with the app's own betting rule.

### What the backtest says (read this before trusting any pick)

| | log-loss |
|---|---|
| closing market alone | 0.5985 |
| best model (+qb) | 0.6010 |
| full model (+injuries) | 0.6044 |

**No feature set beats the closing line.** Rest, weather, divisional games, QB changes
and the official injury report are all already in the price by close. Simulated betting
with the app's rule at closing odds is flat to negative in every configuration. That is
why the default model weight is 10%, not 30%.

What this does *not* test: betting **before** the line moves. The only edge a retail
bettor can realistically claim is timing (injury news minutes before the book adjusts),
and the only way to measure it is CLV in the Bet Log. The 50-graded-bet gate stands.

Also: model files (`.py`) live in the repo; `data/` (snapshots, backtest JSON) and
`bet_log.csv` are gitignored.

## v4.2 (2026-09-17): pricing-error hunting

- `mispricing.py` → `MISPRICING.md`: 34 **pre-registered** market-bias hypotheses (home
  dogs, divisional dogs, unders in wind, big-favourite fades, rest edges, primetime…) tested
  on every closing line 1999–2026. Discovery ≤ 2017 with Benjamini–Hochberg false-discovery
  control (q = 0.10), then out-of-sample validation 2018+. **Result: zero validated pricing
  errors.** One watch-list item: *under in 15+ mph wind* (+7% ROI discovery, +8% validation,
  just misses the FDR bar). The favourite–longshot check shows big dogs (< 20%) lose 10–18%
  at the close: never bet them without a reason.
- `scanner.py` + **🔎 Scanner** mode: live cross-book scan (every US book, h2h / spreads /
  totals). Consensus = mean no-vig probability per line; flags any book ≥ 1.5% off it, true
  arbs (hold < 0), and cross-book spread middles. Also lists this week's games that trigger a
  watch-list rule. Costs 3 API credits per refresh, cached 30 min.
- `promo_engine.py` + **🎁 Promo Hedger** mode: matched-betting calculator for DraftKings
  promos (bonus bet, no-sweat, profit boost, odds boost, qualifying "bet $5 get $150"). Finds
  the DK line with the best hedge at another book on the *same* line, sizes the hedge for
  equal profit either way, prints the two tickets. Manual calculator included.

What "wins regardless of outcome" really means: a fully hedged promo is locked profit, not
a probability. Typical conversions: bonus bets 65–80% of face at long odds, no-sweat bets
~+20–30% of stake, "bet $5 get $150" ≈ +$105. The risks are operational (the hedge line
moving between your two clicks, voids, limits) and DraftKings restricting accounts that only
ever bet promos — spread real bets in, and never hedge at DK itself.
