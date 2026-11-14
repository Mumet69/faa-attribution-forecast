# FAA Attribution Forecaster — Round 2

Entry for make.fun `polymarket-faa-attribution-02`: forecast Polymarket's
FAA outage attribution prices minute by minute, 4 hours ahead.

## Model

Python 3.13, stdlib only, deterministic. For each market, `main.py`
forecasts 5 quantiles (5/25/50/75/95%) for each of the 240 minutes after
the checkpoint, from empirical h-step return quantiles over the last
480 minutes at 15 horizons, interpolated in log(h):

- median recentered on the last price (persistence; drift extrapolation
  was tested and hurts), k = 0.45 mid / 0.9 tails, upside ×0.75 /
  downside ×1.5, mean-based volatility regime, tail shrinkage 0.2,
  constant jump reversion (frac 0.3);
- sum normalisation toward **S0**, the in-case sum of last prices
  (NOT 1.0 — the 4 FAA markets leave a large "none" mass; the live
  sum is ~0.095, so pulling to 1.0 would multiply every median ~10×).

## Round-2 changes (practice mean 0.278826 → 0.264150, −5.3%)

1. **Parametric tail blend (0.75):** empirical quantiles blended 75%
   toward normal quantiles centred on the empirical median, sigma from
   the empirical IQR / 1.349. The 5%/95% empirical quantiles from 480
   overlapping samples are noisy; the blend stabilises them.
2. **Recency weighting:** return history weighted by 0.5^(age/300 min),
   so quantiles adapt faster to regime changes.
3. **Scale-aware jump detection:** a move counts as a jump if large
   absolutely (>0.02) OR large relative to the price (>25% of it), so
   jump reversion actually fires on the tiny FAA prices (0.006–0.037),
   where a −50% crash is only ~0.01 in absolute terms.
4. Small median trend term (weight 0.1 on the last-30-min move, capped
   at h=30) and a wider volatility-regime clip [0.6, 1.8].

## Running

```
python3 main.py   # reads $MAKEFUN_INPUT_DIR (default /input),
                  # writes one <id>.out per <id>.in to $MAKEFUN_OUTPUT_DIR
```

Local practice score (6 NVIDIA practice cases, local scorer):
**mean pinball 0.264150** (p00 0.444954, p01 0.293494, p02 0.351577,
p03 0.244675, p04 0.206793, p05 0.104588), ~0.25–0.35 s/case.
