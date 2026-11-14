#!/usr/bin/env python3
"""FAA attribution forecaster — round-2 final model.

Empirical-quantile forecaster in the NVIDIA round-3 design, improved in
round 2 with two principled changes that together cut practice pinball
loss from 0.278826 to 0.264150 (local scorer, 6 NVIDIA practice cases):
  1. Parametric tail blend: empirical h-step return quantiles are blended
     75% toward normal quantiles centred on the empirical median with
     sigma from the empirical IQR. The 5%/95% empirical quantiles from
     480 overlapping samples are noisy; the blend stabilises them.
  2. Exponential recency weighting (half-life 300 min) of the return
     history, so the empirical quantiles adapt faster to regime changes.
Plus three smaller, practice-confirmed refinements:
  - a small median trend term (weight 0.1 on the last-30-min move, capped
    at h=30): helps jumpy cases, never dominates the persistence median;
  - wider volatility-regime clip [0.6, 1.8] so intervals widen more in
    volatile regimes (the FAA final is jumpier than practice);
  - scale-aware jump detection: a move counts as a jump if it is large
    absolutely (>0.02) OR large relative to the price (>25% of it), so
    jump reversion actually fires on the tiny FAA prices (0.006-0.037),
    where a -50% crash is only ~0.01 in absolute terms.

Sum normalisation still targets S0 = the in-case sum of last prices (NOT
1.0: the 4 FAA markets leave a large "none" mass; live sum is ~0.095).
Median stays a persistence forecast (drift extrapolation was tested and
clearly hurts: +0.012 to +0.092 loss). No network, no randomness.

Stdlib only. Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR (else /output), one .out per .in file, deterministic.
"""
import bisect
import json
import math
import os
import sys

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
GRID = [1, 2, 3, 4, 6, 9, 13, 19, 28, 41, 60, 87, 126, 183, 240]
LOG_GRID = [math.log(h) for h in GRID]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
WINDOW = 480

# Round-2 final parameters (validated on practice).
K_MID = 0.45
K_TAIL = 0.9
UP = 0.75
DOWN = 1.5
VOL_WIN = 60
VOL_LO = 0.6
VOL_HI = 1.8
SHRINK = 0.2
JUMP_THRESH = 0.02
JUMP_REL = 0.25
JUMP_FRAC = 0.3
NORM_A = 0.7
NORM_RAMP = 120.0
TREND_W = 0.1
TREND_WIN = 30
BLEND = 0.75
EXP_HALF = 300.0


def quantile_sorted(s, tau):
    """Linear-interpolation quantile of an already-sorted list."""
    n = len(s)
    if n == 0:
        return 0.0
    if n == 1:
        return s[0]
    pos = tau * (n - 1)
    lo = int(pos)
    hi = lo + 1
    if hi >= n:
        return s[-1]
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def quantile_weighted(pairs, tau):
    """Weighted linear-interpolation quantile. pairs = [(value, weight)]."""
    if not pairs:
        return 0.0
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return pairs[len(pairs) // 2][0]
    target = tau * tot
    acc = 0.0
    for i, (v, w) in enumerate(pairs):
        prev = acc
        acc += w
        if acc >= target:
            if i == 0:
                return v
            pv = pairs[i - 1][0]
            f = (target - prev) / w if w > 0 else 0.0
            return pv * (1.0 - f) + v * f
    return pairs[-1][0]


def interp_log_h(xs, ys, h):
    """Linear interpolation of ys (at grid horizons xs, log scale) at h."""
    lh = math.log(h)
    if lh <= xs[0]:
        return ys[0]
    if lh >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, lh) - 1
    w = (lh - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] * (1.0 - w) + ys[i + 1] * w


def baseline_rows(x):
    rows = []
    for h in range(1, 241):
        s = 0.0025 * math.sqrt(h)
        row = [min(1.0, max(0.0, x + z * s)) for z in Z]
        rows.append([round(v, 6) for v in sorted(row)])
    return rows


def forecast_market(prices):
    """Return 240 rows x 5 quantiles for one market."""
    x = prices[-1]
    n = len(prices)
    if n < 10:
        return baseline_rows(x)
    W = min(WINDOW, n - 1)
    # Empirical h-step return quantiles at grid horizons, endpoints in last
    # W min, with exponential recency weighting (half-life EXP_HALF).
    emp = []  # emp[g][tau_idx]
    for H in GRID:
        pairs = []
        for t in range(n - W, n):
            if t - H >= 0:
                age = (n - 1) - t
                pairs.append((prices[t] - prices[t - H], 0.5 ** (age / EXP_HALF)))
        if len(pairs) < 20:
            pairs = []
            for t in range(n - W, n):
                age = (n - 1) - t
                pairs.append(((prices[t] - prices[t - 1]) * math.sqrt(H),
                              0.5 ** (age / EXP_HALF)))
        emp.append([quantile_weighted(pairs, tau) for tau in TAUS])
    # Blend each grid row's empirical quantiles 75% toward normal quantiles
    # centred on the empirical median, sigma from the empirical IQR.
    for g in range(len(GRID)):
        row = emp[g]
        sig = (row[3] - row[1]) / 1.349 if row[3] > row[1] else 0.0
        for ti in range(5):
            nq = row[2] + Z[ti] * sig
            emp[g][ti] = (1.0 - BLEND) * row[ti] + BLEND * nq

    # Volatility regime: recent mean |1-min move| vs baseline mean |1-min move|.
    base_moves = [abs(prices[t] - prices[t - 1]) for t in range(n - W, n)]
    base = sum(base_moves) / len(base_moves) if base_moves else 0.0
    rw = min(VOL_WIN, n - 1)
    rec_moves = [abs(prices[t] - prices[t - 1]) for t in range(n - rw, n)]
    rec = sum(rec_moves) / len(rec_moves) if rec_moves else 0.0
    regime = 1.0
    if base > 1e-12:
        regime = min(VOL_HI, max(VOL_LO, rec / base))

    # Jump reversion (constant): last 1-min move, if large absolutely
    # (>0.02) or relative to the price (>25% of it).
    jump = 0.0
    d = prices[-1] - prices[-2]
    thresh = min(JUMP_THRESH, JUMP_REL * max(x, 1e-6))
    if abs(d) > thresh:
        jump = JUMP_FRAC * d

    # Recent trend (last TREND_WIN minutes) for a small median drift.
    trend = prices[-1] - prices[-1 - TREND_WIN] if n > TREND_WIN else 0.0

    rows = []
    for h in range(1, 241):
        # Interpolate each tau's empirical quantile in log(h).
        eq = [interp_log_h(LOG_GRID, [emp[g][ti] for g in range(len(GRID))], h)
              for ti in range(5)]
        med = eq[2]
        centre = TREND_W * trend * min(h, TREND_WIN) / TREND_WIN
        row = []
        for ti, tau in enumerate(TAUS):
            k = K_MID if tau in (0.25, 0.5, 0.75) else K_TAIL
            dev = k * (eq[ti] - med)
            dev = dev * UP if dev > 0 else dev * DOWN
            dev *= regime
            if tau in (0.05, 0.95):
                dev *= (1.0 - SHRINK)
            row.append(x + centre + dev)
        if jump != 0.0:
            row = [v - jump for v in row]
        rows.append(row)
    return rows


def forecast_case(case):
    markets = case["markets"]
    per_market = []
    xs = []
    for m in markets:
        per_market.append(forecast_market(m["prices"]))
        xs.append(m["prices"][-1])
    # Sum normalisation toward S0 (in-case sum of last prices).
    if len(markets) > 1:
        target = sum(xs)
        for h in range(240):
            s = sum(per_market[mi][h][2] for mi in range(len(markets)))
            if s <= 1e-9:
                continue
            ramp = min(1.0, (h + 1) / NORM_RAMP)
            for mi in range(len(markets)):
                delta = NORM_A * xs[mi] * (target / s - 1.0) * ramp
                per_market[mi][h] = [v + delta for v in per_market[mi][h]]
    out = []
    for m, rows in zip(markets, per_market):
        clean = []
        for row in rows:
            r = [min(1.0, max(0.0, v)) for v in row]
            r = sorted(r)
            clean.append([round(v, 6) for v in r])
        out.append({"id": m["id"], "q": clean})
    return {"forecasts": out}


def main():
    indir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    outdir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(outdir, exist_ok=True)
    for fn in sorted(os.listdir(indir)):
        if not fn.endswith(".in"):
            continue
        with open(os.path.join(indir, fn)) as f:
            case = json.load(f)
        res = forecast_case(case)
        with open(os.path.join(outdir, fn[:-3] + ".out"), "w") as f:
            json.dump(res, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
