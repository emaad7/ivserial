"""Command line interface.

Examples
--------
  ivserial data.csv                          # test every numeric column
  ivserial prices.csv --column Close --transform logdiff --seed 1
  ivserial data.xlsx --B 99999 --out report.txt --json results.json
  ivserial plan --gamma 0.05 --m 3 --d 4     # sample size (or --v 0.06; --n 500 for power)
  ivserial null --m 3 --d 4 --q 80           # null limit, critical values, P-value
  ivserial gui                               # graphical interface
  ivserial guide                             # open the user guide in the browser
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

from . import __version__
from .core import serial_test
from .io import read_table


def _benjamini_hochberg(p):
    import numpy as np
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def open_guide():
    """Open the illustrated user guide (GUIDE.html, installed with the package) in the browser."""
    import os
    import webbrowser
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GUIDE.html")
    webbrowser.open("file://" + path)
    print(path)


def _plan(argv):
    from .core import approximate_power, choose_resolution, sample_size_table
    ap = argparse.ArgumentParser(prog="ivserial plan", description="Sample size and power (Corollary 4.8).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gamma", type=float, help="strength of dependence Gamma at (m, d)")
    g.add_argument("--v", type=float, help="serial Cramer coefficient V at (m, d), instead of Gamma")
    ap.add_argument("--tau", type=float, default=0.0, help="asymptotic standard deviation (0: weak dependence)")
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, nargs="*", default=[0.5, 0.8, 0.9, 0.95])
    ap.add_argument("--n", type=int, help="also report the power for this number of observations")
    a = ap.parse_args(argv)
    if a.gamma is None:
        from .core import gamma_from_cramer_v
        a.gamma = gamma_from_cramer_v(a.v, a.m, a.d)
        print(f"V = {a.v} at (m, d) = ({a.m}, {a.d}) corresponds to Gamma = {a.gamma:.4f}")
    print(f"Required number of observations (Gamma = {a.gamma:.4g}, tau = {a.tau}, (m, d) = ({a.m}, {a.d}), level {a.alpha})")
    print(f"{'power':>8}{'n':>10}")
    for p, n in sample_size_table(a.gamma, a.m, a.d, a.alpha, a.tau, a.power):
        print(f"{p:>8.2f}{n:>10d}")
    if a.n:
        print(f"Approximate power with n = {a.n}: {approximate_power(a.gamma, a.n, a.m, a.d, a.alpha, a.tau):.3f}")
        m, d = choose_resolution(a.n)
        print(f"Default resolution for n = {a.n}: m = {m}, d = {d}")
    return 0


def _null(argv):
    from .core import null_limit_sf, null_limit_summary
    ap = argparse.ArgumentParser(prog="ivserial null", description="Null limit L_{m,d} of Q (Theorem 4.1).")
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--q", type=float, help="value of Q: report P(L > q)")
    a = ap.parse_args(argv)
    s = null_limit_summary(a.m, a.d)
    print("L = " + " + ".join(f"{int(w)} x chi2({df})" for w, df in s["components"]))
    print(f"mean = {s['mean']:.2f}, variance = {s['variance']:.2f}")
    for lev, c in s["critical"].items():
        print(f"critical value at level {lev:.2f}: {c:.3f}")
    if a.q is not None:
        print(f"P(L > {a.q:g}) = {null_limit_sf(a.q, a.m, a.d):.5f}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "plan":
        return _plan(argv[1:])
    if argv and argv[0] == "null":
        return _null(argv[1:])
    if argv and argv[0] == "guide":
        open_guide()
        return 0
    if argv and argv[0] == "gui":
        from .gui import main as gui_main
        gui_main()
        return 0
    ap = argparse.ArgumentParser(prog="ivserial", description="Exact symbolic test of serial independence.")
    ap.add_argument("file", help="CSV, TXT or Excel file; one series per column, in time order")
    ap.add_argument("--column", "-c", action="append", help="column(s) to test (default: all numeric columns)")
    ap.add_argument("--transform", "-t", choices=["none", "diff", "logdiff"], default="none",
                    help="apply differences or log differences (log returns, %%) first; use logdiff for prices")
    ap.add_argument("--B", type=int, default=9999, help="number of random permutations (default 9999)")
    ap.add_argument("--seed", type=int, default=None, help="random seed for reproducible P-values")
    ap.add_argument("--m", type=int, default=None, help="block length (default: automatic)")
    ap.add_argument("--d", type=int, default=None, help="number of classes (default: automatic)")
    ap.add_argument("--alpha", type=float, default=0.05, help="level of the reported decision")
    ap.add_argument("--lags", type=int, default=2, help="portmanteau lags (default 2)")
    ap.add_argument("--d2", type=int, default=None, help="portmanteau classes (default: automatic)")
    ap.add_argument("--cells", type=int, default=8, help="number of patterns shown (default 8)")
    ap.add_argument("--out", help="write the report to this text file")
    ap.add_argument("--json", help="write all results to this JSON file")
    ap.add_argument("--version", action="version", version=f"ivserial {__version__}")
    a = ap.parse_args(argv)

    cols = read_table(a.file)
    names = a.column or list(cols)
    missing = [c for c in names if c not in cols]
    if missing:
        ap.error(f"column(s) not found or not numeric: {missing}; available: {list(cols)}")

    results, reports = [], []
    for name in names:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                r = serial_test(cols[name], m=a.m, d=a.d, d_portmanteau=a.d2, lags=a.lags, B=a.B,
                                seed=a.seed, alpha=a.alpha, n_cells=max(a.cells, 1),
                                transform=None if a.transform == "none" else a.transform, label=name)
            except ValueError as e:
                reports.append(f"[{name}] skipped: {e}")
                continue
        results.append(r)
        rep = r.summary(n_cells=a.cells)
        if w:
            rep += "\nNote: " + "; ".join(str(x.message) for x in w)
        reports.append(rep)

    if len(results) > 1:
        q = _benjamini_hochberg([r.p_C for r in results])
        lines = ["", "Screening summary (Benjamini-Hochberg adjustment over the series tested)",
                 f"{'series':<24}{'n':>7}{'p_C':>10}{'adj. p':>10}{'V-hat':>8}{'Gamma-hat':>11}  decision"]
        for r, qq in zip(results, q):
            lines.append(f"{r.label[:23]:<24}{r.n:>7d}{r.p_C:>10.4f}{qq:>10.4f}{r.cramer_v:>8.3f}{r.gamma_hat:>11.4f}  "
                         f"{'reject' if qq <= a.alpha else '-'}")
        reports.append("\n".join(lines))

    text = "\n\n".join(reports)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
