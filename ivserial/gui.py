"""Graphical interface (tkinter, included with Python).

Start with:  ivserial-gui,  python -m ivserial,  or double-click  ivserial_gui.pyw

Three tabs:
  Test                - load a file, choose series and options, run the exact test
  Sample size & power - plan a study (Corollary 4.8), optionally from the last test
  Null distribution   - the explicit limit L_{m,d}: moments, critical values, P-values
"""
from __future__ import annotations

import json
import os
import threading
import warnings

from . import __version__
from .core import (approximate_power, choose_resolution, gamma_from_cramer_v, null_limit_sf,
                   null_limit_summary, sample_size_table, serial_cramer_v, serial_test)
from .io import read_table

HELP = (
    "TEST tab\n"
    "1. Open a CSV, TXT or Excel file: one series per column, rows in time order.\n"
    "2. Choose the column, or tick 'test all columns' to screen every column.\n"
    "3. For prices choose 'log returns'.\n"
    "4. Press Run test. Save report writes a text or JSON file.\n\n"
    "C is the proposed test. A P-value below the level means that the series is not a sequence "
    "of independent, identically distributed values. V-hat (0 to 1) measures the strength of the "
    "dependence and Gamma-hat the efficiency; the pattern table shows which sequences of classes occur more or less often "
    "than under independence.\n\n"
    "SAMPLE SIZE & POWER tab\n"
    "Enter the strength of dependence Gamma (or press 'Use last test') to obtain the number of "
    "observations needed for a given power, or the power for a given number of observations.\n\n"
    "NULL DISTRIBUTION tab\n"
    "Mean, variance and critical values of the limit L_{m,d}, and the P-value of any value of Q.\n\n"
    "Full illustrated guide: press User guide (or run: ivserial guide).")


def main():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText

    root = tk.Tk()
    root.title(f"ivserial {__version__} - exact symbolic test of serial independence")
    root.geometry("980x720")
    state = {"cols": {}, "path": None, "results": []}

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=6, pady=6)
    tab_test, tab_plan, tab_null = (ttk.Frame(nb, padding=10) for _ in range(3))
    nb.add(tab_test, text="  Test  ")
    nb.add(tab_plan, text="  Sample size & power  ")
    nb.add(tab_null, text="  Null distribution  ")

    def entry(parent, label, var, width=8):
        ttk.Label(parent, text=label).pack(side="left")
        ttk.Entry(parent, textvariable=var, width=width).pack(side="left", padx=(4, 14))

    def parse_int(s, name, allow_auto=True):
        s = s.strip().lower()
        if allow_auto and s in ("", "auto", "none"):
            return None
        try:
            return int(s)
        except ValueError:
            raise ValueError(f"{name} must be an integer" + (" or 'auto'" if allow_auto else ""))

    def parse_float(s, name):
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"{name} must be a number")

    def output_box(parent):
        box = ScrolledText(parent, font=("Courier", 10), wrap="none")
        box.pack(fill="both", expand=True, pady=(10, 0))
        return box

    def write(box, text):
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)

    # ======================================================================
    # Tab 1: Test
    # ======================================================================
    top = ttk.Frame(tab_test)
    top.pack(fill="x")
    file_var = tk.StringVar(value="No file selected")
    ttk.Button(top, text="Open data file...", command=lambda: open_file()).grid(row=0, column=0, sticky="w")
    ttk.Label(top, textvariable=file_var, foreground="#444").grid(row=0, column=1, sticky="w", padx=8)

    ttk.Label(top, text="Series (column):").grid(row=1, column=0, sticky="w", pady=(10, 0))
    colf = ttk.Frame(top)
    colf.grid(row=1, column=1, sticky="w", pady=(10, 0))
    col_var = tk.StringVar()
    col_box = ttk.Combobox(colf, textvariable=col_var, state="readonly", width=30)
    col_box.pack(side="left")
    all_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(colf, text="test all columns (screening)", variable=all_var).pack(side="left", padx=12)

    ttk.Label(top, text="Transform:").grid(row=2, column=0, sticky="w", pady=(8, 0))
    tr_var = tk.StringVar(value="none")
    trf = ttk.Frame(top)
    trf.grid(row=2, column=1, sticky="w", pady=(8, 0))
    for txt, val in (("none", "none"), ("differences", "diff"), ("log returns (prices)", "logdiff")):
        ttk.Radiobutton(trf, text=txt, variable=tr_var, value=val).pack(side="left", padx=(0, 12))

    ttk.Label(top, text="Options:").grid(row=3, column=0, sticky="w", pady=(8, 0))
    opt = ttk.Frame(top)
    opt.grid(row=3, column=1, sticky="w", pady=(8, 0))
    B_var, seed_var, m_var, d_var, a_var = (tk.StringVar(value=v) for v in ("9999", "1", "auto", "auto", "0.05"))
    entry(opt, "Permutations B", B_var)
    entry(opt, "Seed", seed_var, 6)
    entry(opt, "Block length m", m_var, 6)
    entry(opt, "Classes d", d_var, 6)
    entry(opt, "Level", a_var, 6)

    ttk.Label(top, text="Portmanteau:").grid(row=4, column=0, sticky="w", pady=(8, 0))
    opt2 = ttk.Frame(top)
    opt2.grid(row=4, column=1, sticky="w", pady=(8, 0))
    lag_var, d2_var, nc_var = (tk.StringVar(value=v) for v in ("2", "auto", "8"))
    entry(opt2, "Lags", lag_var, 6)
    entry(opt2, "Classes", d2_var, 6)
    entry(opt2, "Patterns shown", nc_var, 6)

    btn = ttk.Frame(tab_test)
    btn.pack(fill="x", pady=(12, 0))
    run_btn = ttk.Button(btn, text="Run test", command=lambda: run())
    run_btn.pack(side="left")
    ttk.Button(btn, text="Save report...", command=lambda: save()).pack(side="left", padx=8)
    ttk.Button(btn, text="Help", command=lambda: messagebox.showinfo("How to use", HELP)).pack(side="left")

    def _guide():
        from .cli import open_guide
        open_guide()
    ttk.Button(btn, text="User guide", command=_guide).pack(side="left", padx=8)
    status = tk.StringVar(value="Open a CSV, TXT or Excel file with one series per column, in time order.")
    ttk.Label(btn, textvariable=status, foreground="#225").pack(side="left", padx=14)
    out = output_box(tab_test)

    def open_file():
        path = filedialog.askopenfilename(title="Open data file",
                                          filetypes=[("Data files", "*.csv *.txt *.dat *.xlsx *.xls"),
                                                     ("All files", "*.*")])
        if not path:
            return
        try:
            cols = read_table(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Cannot read file", str(e))
            return
        state.update(cols=cols, path=path)
        file_var.set(f"{os.path.basename(path)}   ({len(cols)} numeric column(s))")
        col_box["values"] = list(cols)
        col_box.current(0)
        status.set("Choose the series and options, then press Run test.")

    def run():
        if not state["cols"]:
            messagebox.showinfo("No data", "Open a data file first.")
            return
        try:
            B = parse_int(B_var.get(), "B") or 9999
            seed = parse_int(seed_var.get(), "Seed")
            m = parse_int(m_var.get(), "m")
            d = parse_int(d_var.get(), "d")
            lags = parse_int(lag_var.get(), "Lags", False)
            d2 = parse_int(d2_var.get(), "Portmanteau classes")
            ncell = parse_int(nc_var.get(), "Patterns shown", False)
            alpha = parse_float(a_var.get(), "Level")
        except ValueError as e:
            messagebox.showerror("Invalid option", str(e))
            return
        names = list(state["cols"]) if all_var.get() else [col_var.get()]
        tr = None if tr_var.get() == "none" else tr_var.get()
        run_btn.configure(state="disabled")
        status.set("Running ...")

        def work():
            reports, results = [], []
            for name in names:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    try:
                        r = serial_test(state["cols"][name], m=m, d=d, d_portmanteau=d2, lags=lags, B=B,
                                        seed=seed, alpha=alpha, transform=tr, label=name,
                                        n_cells=max(ncell, 1))
                    except Exception as e:  # noqa: BLE001
                        reports.append(f"[{name}] not tested: {e}")
                        continue
                results.append(r)
                rep = r.summary(n_cells=ncell)
                if w:
                    rep += "\nNote: " + "; ".join(str(x.message) for x in w)
                reports.append(rep)
            if len(results) > 1:
                from .cli import _benjamini_hochberg
                q = _benjamini_hochberg([r.p_C for r in results])
                lines = ["Screening summary (Benjamini-Hochberg adjustment)",
                         f"{'series':<24}{'n':>7}{'p_C':>10}{'adj. p':>10}{'V-hat':>8}{'Gamma-hat':>11}  decision"]
                for r, qq in zip(results, q):
                    lines.append(f"{r.label[:23]:<24}{r.n:>7d}{r.p_C:>10.4f}{qq:>10.4f}{r.cramer_v:>8.3f}{r.gamma_hat:>11.4f}  "
                                 f"{'reject' if qq <= alpha else '-'}")
                reports.insert(0, "\n".join(lines))
            text = "\n\n".join(reports)

            def done():
                state["results"] = results
                write(out, text)
                run_btn.configure(state="normal")
                status.set("Done." if results else "No series could be tested.")
            root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def save():
        text = out.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Nothing to save", "Run the test first.")
            return
        path = filedialog.asksaveasfilename(title="Save report", defaultextension=".txt",
                                            filetypes=[("Text report", "*.txt"), ("JSON results", "*.json")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            if path.lower().endswith(".json"):
                json.dump([r.to_dict() for r in state["results"]], f, indent=1)
            else:
                f.write(text + "\n")
        status.set(f"Saved to {os.path.basename(path)}")

    # ======================================================================
    # Tab 2: Sample size & power
    # ======================================================================
    ttk.Label(tab_plan, wraplength=920, justify="left", text=(
        "Plan a study with Corollary 4.8 of the paper. Gamma is the strength of dependence "
        "(normalized information variance) at resolution (m, d); instead of Gamma you may enter the serial "
        "Cramer coefficient V (between 0 and 1). tau is the asymptotic standard "
        "deviation of the statistic under the alternative. tau = 0 is accurate for weak dependence "
        "and otherwise optimistic. 'Use last test' takes Gamma-hat, tau-hat, m and d from the most "
        "recent test, e.g. a pilot study or a comparable series.")).pack(anchor="w")
    pf = ttk.Frame(tab_plan)
    pf.pack(fill="x", pady=(10, 0))
    g_var, t_var, pm_var, pd_var, pa_var, pp_var, pn_var = (
        tk.StringVar(value=v) for v in ("0.05", "0", "3", "4", "0.05", "0.8", "500"))
    entry(pf, "Gamma", g_var)
    entry(pf, "tau", t_var)
    entry(pf, "m", pm_var, 5)
    entry(pf, "d", pd_var, 5)
    entry(pf, "Level", pa_var, 6)
    pf2 = ttk.Frame(tab_plan)
    pf2.pack(fill="x", pady=(8, 0))
    entry(pf2, "Target power", pp_var, 6)
    entry(pf2, "or: number of observations n", pn_var, 8)
    v_var = tk.StringVar(value="")
    entry(pf2, "V (optional, instead of Gamma)", v_var, 6)
    pb = ttk.Frame(tab_plan)
    pb.pack(fill="x", pady=(10, 0))
    ttk.Button(pb, text="Use last test", command=lambda: use_last()).pack(side="left")
    ttk.Button(pb, text="Required sample size", command=lambda: plan_n()).pack(side="left", padx=8)
    ttk.Button(pb, text="Power for n", command=lambda: plan_power()).pack(side="left")
    ttk.Button(pb, text="Resolution for n", command=lambda: plan_res()).pack(side="left", padx=8)
    plan_out = output_box(tab_plan)

    def plan_params():
        m = parse_int(pm_var.get(), "m", False)
        d = parse_int(pd_var.get(), "d", False)
        if v_var.get().strip():
            g = gamma_from_cramer_v(parse_float(v_var.get(), "V"), m, d)
            g_var.set(f"{g:.4f}")
            v_var.set("")
        return (parse_float(g_var.get(), "Gamma"), parse_float(t_var.get(), "tau"), m, d,
                parse_float(pa_var.get(), "Level"))

    def use_last():
        if not state["results"]:
            messagebox.showinfo("No test yet", "Run a test in the Test tab first.")
            return
        r = state["results"][-1]
        g_var.set(f"{max(r.gamma_hat, 0):.4f}")
        t_var.set(f"{r.tau_hat:.4f}")
        pm_var.set(str(r.m))
        pd_var.set(str(r.d))
        write(plan_out, f"Taken from the last test ({r.label or 'series'}, n = {r.n}): V-hat = {r.cramer_v:.3f}, Gamma-hat = {r.gamma_hat:.4f}, "
                        f"tau-hat = {r.tau_hat:.4f}, m = {r.m}, d = {r.d}.")

    def plan_n():
        try:
            g, tau, m, d, a = plan_params()
            if g <= 0:
                raise ValueError("Gamma must be positive")
            target = parse_float(pp_var.get(), "Target power")
            powers = sorted(set([0.5, 0.8, 0.9, 0.95, target]))
            rows = sample_size_table(g, m, d, a, tau, powers)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Invalid input", str(e))
            return
        L = [f"Required number of observations (Gamma = {g}, V = {serial_cramer_v(g, m, d):.3f}, tau = {tau}, "
             f"(m, d) = ({m}, {d}), level {a})",
             f"{'power':>8}{'n':>10}"]
        for p, n in rows:
            L.append(f"{p:>8.2f}{n:>10d}" + ("   <- target" if abs(p - target) < 1e-12 else ""))
        L.append("")
        L.append("Note: Gamma depends on the resolution (m, d); use values measured at the same (m, d).")
        write(plan_out, "\n".join(L))

    def plan_power():
        try:
            g, tau, m, d, a = plan_params()
            n = parse_int(pn_var.get(), "n", False)
            pw = approximate_power(g, n, m, d, a, tau)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Invalid input", str(e))
            return
        write(plan_out, f"Approximate power with n = {n} observations: {pw:.3f}\n"
                        f"(Gamma = {g}, tau = {tau}, (m, d) = ({m}, {d}), level {a})")

    def plan_res():
        try:
            n = parse_int(pn_var.get(), "n", False)
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        m, d = choose_resolution(n)
        write(plan_out, f"For n = {n} the default resolution is m = {m}, d = {d} ({d ** m} patterns, "
                        f"{(n - m + 1) / d ** m:.1f} blocks per pattern on average).")
        pm_var.set(str(m))
        pd_var.set(str(d))

    # ======================================================================
    # Tab 3: Null distribution
    # ======================================================================
    ttk.Label(tab_null, wraplength=920, justify="left", text=(
        "Under independence, the block statistic Q converges to L_{m,d} = sum over s = 2..m of "
        "(m - s + 1) times a chi-square variable with (d - 1)^2 d^(s - 2) degrees of freedom "
        "(Theorem 4.1). The permutation test of the Test tab is exact; this tab gives the explicit "
        "limit, accurate for n >= 350.")).pack(anchor="w")
    nf = ttk.Frame(tab_null)
    nf.pack(fill="x", pady=(10, 0))
    nm_var, nd_var, nq_var = (tk.StringVar(value=v) for v in ("3", "4", "75"))
    entry(nf, "m", nm_var, 5)
    entry(nf, "d", nd_var, 5)
    entry(nf, "Value of Q (optional)", nq_var, 10)
    ttk.Button(nf, text="Compute", command=lambda: null_calc()).pack(side="left", padx=8)
    null_out = output_box(tab_null)

    def null_calc():
        try:
            m = parse_int(nm_var.get(), "m", False)
            d = parse_int(nd_var.get(), "d", False)
            s = null_limit_summary(m, d)
            q = nq_var.get().strip()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Invalid input", str(e))
            return
        comp = " + ".join(f"{int(w)} x chi2({df})" for w, df in s["components"])
        L = [f"Null limit for m = {m}, d = {d} ({s['patterns']} patterns):  L = {comp}",
             f"mean = {s['mean']:.2f}, variance = {s['variance']:.2f}", "",
             f"{'level':>8}{'critical value':>18}"]
        for a, c in s["critical"].items():
            L.append(f"{a:>8.2f}{c:>18.3f}")
        if q:
            try:
                qv = float(q)
                L += ["", f"P(L > {qv:g}) = {null_limit_sf(qv, m, d):.5f}"]
            except ValueError:
                L += ["", "The value of Q must be a number."]
        write(null_out, "\n".join(L))

    root.mainloop()


if __name__ == "__main__":
    main()
