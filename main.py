#!/usr/bin/env python3
"""FAA attribution forecaster (model prototype).

Empirical-quantile forecaster in the NVIDIA round-3 design, re-implemented
from the design description (no copied code):
  q_tau(h) = x + k_tau * (emp_q_tau(h) - emp_q_50(h))
with k=0.45 mid / 0.9 tails, upside x0.75 / downside x1.5 asymmetry,
mean-based volatility regime scaling, tail shrinkage, jump reversion
(decaying or constant), and sum normalisation toward S0 = the in-case sum
of last prices (NOT 1.0: the 4 FAA markets leave a large "none" mass).

Stdlib only. Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR (else /output), one .out per .in file, deterministic.

Tuning knobs (env vars, defaults = round-3 values):
  FAA_K_MID FAA_K_TAIL FAA_UP FAA_DOWN
  FAA_VOL (0/1) FAA_VOL_WIN FAA_VOL_LO FAA_VOL_HI
  FAA_SHRINK FAA_JUMP (0/1) FAA_JUMP_THRESH FAA_JUMP_FRAC
  FAA_JUMP_MODE (decay|const) FAA_JUMP_HALF
  FAA_SUMNORM (s0|one|off) FAA_NORM_A FAA_NORM_RAMP
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
WINDOW = int(os.environ.get("FAA_WINDOW", 480))


def envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def envi(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


K_MID = envf("FAA_K_MID", 0.45)
K_TAIL = envf("FAA_K_TAIL", 0.9)
UP = envf("FAA_UP", 0.75)
DOWN = envf("FAA_DOWN", 1.5)
VOL_ON = envi("FAA_VOL", 1)
VOL_WIN = envi("FAA_VOL_WIN", 60)
VOL_LO = envf("FAA_VOL_LO", 0.7)
VOL_HI = envf("FAA_VOL_HI", 1.5)
SHRINK = envf("FAA_SHRINK", 0.2)
JUMP_ON = envi("FAA_JUMP", 1)
JUMP_THRESH = envf("FAA_JUMP_THRESH", 0.02)
JUMP_FRAC = envf("FAA_JUMP_FRAC", 0.3)
JUMP_MODE = os.environ.get("FAA_JUMP_MODE", "const")
JUMP_HALF = envf("FAA_JUMP_HALF", 60.0)
SUMNORM = os.environ.get("FAA_SUMNORM", "s0")
NORM_A = envf("FAA_NORM_A", 0.7)
NORM_RAMP = envf("FAA_NORM_RAMP", 120.0)


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
    # Empirical h-step return quantiles at grid horizons, endpoints in last W min.
    emp = []  # emp[g][tau_idx]
    for H in GRID:
        rets = []
        for t in range(n - W, n):
            if t - H >= 0:
                rets.append(prices[t] - prices[t - H])
        if len(rets) < 20:
            # Not enough history for this horizon: fall back to scaled 1-step.
            rets = [prices[t] - prices[t - 1] for t in range(n - W, n)]
            rets = [r * math.sqrt(H) for r in rets]
        rets.sort()
        emp.append([quantile_sorted(rets, tau) for tau in TAUS])

    # Volatility regime: recent mean |1-min move| vs baseline mean |1-min move|.
    regime = 1.0
    if VOL_ON:
        base_moves = [abs(prices[t] - prices[t - 1]) for t in range(n - W, n)]
        base = sum(base_moves) / len(base_moves) if base_moves else 0.0
        rw = min(VOL_WIN, n - 1)
        rec_moves = [abs(prices[t] - prices[t - 1]) for t in range(n - rw, n)]
        rec = sum(rec_moves) / len(rec_moves) if rec_moves else 0.0
        if base > 1e-12:
            regime = min(VOL_HI, max(VOL_LO, rec / base))

    # Jump reversion: last 1-min move, if large.
    jump = 0.0
    if JUMP_ON:
        d = prices[-1] - prices[-2]
        if abs(d) > JUMP_THRESH:
            jump = JUMP_FRAC * d

    rows = []
    for h in range(1, 241):
        # Interpolate each tau's empirical quantile in log(h).
        eq = [interp_log_h(LOG_GRID, [emp[g][ti] for g in range(len(GRID))], h)
              for ti in range(5)]
        med = eq[2]
        row = []
        for ti, tau in enumerate(TAUS):
            k = K_MID if tau in (0.25, 0.5, 0.75) else K_TAIL
            dev = k * (eq[ti] - med)
            dev = dev * UP if dev > 0 else dev * DOWN
            dev *= regime
            if tau in (0.05, 0.95):
                dev *= (1.0 - SHRINK)
            row.append(x + dev)
        if jump != 0.0:
            if JUMP_MODE == "const":
                rev = jump
            else:  # decay: exp(-h / half-life-ish scale)
                rev = jump * math.exp(-h / JUMP_HALF)
            row = [v - rev for v in row]
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
    if SUMNORM != "off" and len(markets) > 1:
        target = sum(xs) if SUMNORM == "s0" else 1.0
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
