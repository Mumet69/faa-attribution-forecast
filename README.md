# polymarket-faa-attribution-02 — FAA outage attribution forecaster

Entry for make.fun competition `polymarket-faa-attribution-02`: forecast the
Yes price of every market of Polymarket's FAA outage attribution family at
every minute of the 4 hours after each checkpoint, as five quantiles
(5%, 25%, 50%, 75%, 95%) per market per minute.

## How it works

`main.py` (Python 3.13, stdlib only, no network) is an empirical-quantile
forecaster in the NVIDIA round-3 design:

```
q_tau(h) = x + k_tau * (emp_q_tau(h) - emp_q_50(h))
```

- Empirical h-step return quantiles over the last 480 minutes, at horizons
  {1, 2, 3, 4, 6, 9, 13, 19, 28, 41, 60, 87, 126, 183, 240}, interpolated
  in log(h); median recentered to the last price.
- k = 0.45 for middle quantiles, 0.9 for tails; upside deviations x0.75,
  downside x1.5.
- Mean-based volatility regime scaling, tail shrinkage, and jump reversion
  (constant mode) for large last-minute moves.
- Sum normalisation toward S0, the in-case sum of last prices (not 1.0:
  the 4 FAA markets leave a large "none" mass), ramped over 120 minutes.
- All quantiles clamped to [0, 1] and sorted monotone.

## Run

```sh
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output \
  python3 main.py
```

Reads one `<case>.in` per case, writes `<case>.out` per case.
Deterministic: two runs produce byte-identical output.
Practice (6 NVIDIA cases from the kit): mean pinball loss 0.278826.

## License

MIT © serli69 — see LICENSE.
