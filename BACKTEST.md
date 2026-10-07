# Backtest — walk-forward, seasons [2024, 2025, 2026]

Generated 2026-09-17T08:11 · 560 games · trained only on games before each week · scored at the **closing** moneyline.

## Does anything beat the market?

| feature set | log-loss | Brier | acc | bets (raw) | hit | flat ROI | bets (app blend) | hit | flat ROI | ¼-Kelly bank (100 start) |
|---|---|---|---|---|---|---|---|---|---|---|
| **market alone (no model)** | 0.5985 | 0.2061 | — | — | — | — | — | — | — | — |
| market | 0.5989 | 0.2063 | 0.686 | 0 | — | — | 0 | — | — | 100.0 |
| +team | 0.6007 | 0.2071 | 0.686 | 174 | 0.511 | -3.9% | 9 | 0.111 | -70.7% | 96.8 |
| +qb | 0.601 | 0.2074 | 0.68 | 183 | 0.519 | +0.0% | 8 | 0.375 | -5.2% | 98.4 |
| +context | 0.6054 | 0.2093 | 0.686 | 199 | 0.492 | -3.9% | 13 | 0.154 | -55.0% | 95.1 |
| +injuries | 0.6044 | 0.2089 | 0.686 | 218 | 0.523 | -2.2% | 14 | 0.286 | -15.1% | 96.1 |

Lower log-loss / Brier = better probabilities. A feature set earns its place only if it lowers log-loss **below the market row**. Flat ROI is per 1-unit bet at the closing price with the app's 3% min-EV rule.

## By season (full model, app blend)

| season | games | market log-loss | model log-loss | bets | hit | flat ROI | ¼-Kelly bank |
|---|---|---|---|---|---|---|---|
| 2024 | 272 | 0.5875 | 0.5838 | 5 | 0.2 | -37.0% | 97.9 |
| 2025 | 272 | 0.6082 | 0.6238 | 9 | 0.333 | -2.9% | 98.2 |
| 2026 | 16 | 0.6209 | 0.623 | 0 | — | — | 100.0 |

## Calibration (full model)

| predicted home-win bucket | n | mean predicted | actual |
|---|---|---|---|
| (0.0, 0.3] | 85 | 0.23 | 0.212 |
| (0.3, 0.4] | 62 | 0.353 | 0.403 |
| (0.4, 0.5] | 84 | 0.45 | 0.369 |
| (0.5, 0.6] | 96 | 0.555 | 0.573 |
| (0.6, 0.7] | 84 | 0.648 | 0.667 |
| (0.7, 0.8] | 80 | 0.75 | 0.7 |
| (0.8, 1.0] | 69 | 0.864 | 0.87 |

## Standardised coefficients, full model

| feature | coef |
|---|---|
| logit_mkt | +1.009 |
| diff_pass | +0.122 |
| diff_rush | -0.104 |
| diff_qb | -0.157 |
| diff_qb_change | -0.088 |
| rest_diff | -0.115 |
| div_game | +0.074 |
| outdoor | +0.025 |
| wind | -0.021 |
| diff_inj | -0.175 |

Caveats: `wind` is the schedule's recorded game-day wind (a stand-in for the forecast the app uses live). Injury loads use the official Wed–Fri reports, which the closing line has already seen. Team ratings are EWM (half-life 6 games, 60% offseason carry); QB ratings follow the player (half-life 8 games).
