# ivserial — exact symbolic tests of serial independence

`ivserial` tests whether a time series is a sequence of independent and identically
distributed (i.i.d.) values. It implements the procedure of

> Ashtari Nezhad, E. (2026). *Information Variance and Exact Symbolic Tests of Serial Independence.*

The test is **rank based and exact**: under independence its P-value is valid for every
sample size and every continuous distribution, with no model, bandwidth or tuning to choose.
Besides the P-value it reports **how strong** the dependence is and **which patterns** of
consecutive values cause it.

The package can be used in two ways that give identical results: a **window** (GUI, no
programming) or **code** (a Python function or a terminal command). Everything available in
code — the test, screening of many series, sample-size and power planning, and the null
distribution — is also available in the window.

**Full illustrated user guide: [`GUIDE.html`](GUIDE.html)** — open it in any web browser, or run `ivserial guide` after installation (also the *User guide* button in the window).

---

## 1. Installation

You need **Python 3.8 or later** (free, from <https://www.python.org/downloads/>; on Windows tick
**"Add python.exe to PATH"** in the installer). Then install the package like any other module:

```
pip install ivserial                                   # from PyPI (recommended)
pip install git+https://github.com/emaad7/ivserial    # latest version from GitHub (needs Git)
```

PyPI: <https://pypi.org/project/ivserial/> · Source: <https://github.com/emaad7/ivserial>.
Update with `pip install --upgrade ivserial`. Offline: download the wheel
`ivserial-1.0.1-py3-none-any.whl` (GitHub *Releases* or PyPI *Download files*) and run
`pip install ivserial-1.0.1-py3-none-any.whl`.

`numpy` and `scipy` are installed automatically; for Excel files add `pip install pandas openpyxl`.
If `pip` is not found, use `python -m pip`. This gives the module `ivserial`, the command
`ivserial`, and the command `ivserial-gui`, which opens the window. Check with `ivserial --version`.

Without installing: on GitHub press *Code → Download ZIP*, unzip, run `pip install numpy scipy`, and double-click `run_gui_windows.bat`
(Windows), `run_gui_mac_linux.command` (macOS/Linux) or `ivserial_gui.pyw`.

---

## 2. Your data

* A **CSV**, **TXT** or **Excel** (`.xlsx`) file.
* **One series per column, rows in time order** (oldest first). A first row with column
  names is optional; non-numeric columns, such as dates, are ignored.
* Missing values are removed.
* At least 20 observations; 100 or more are recommended, and the test is most informative
  for several hundred.

`examples/exchange_rates.csv` contains the 20 daily exchange rates used in the paper
(prices; use the *log returns* transform).

**Prices or returns?** The test is for the values you give it. For prices, levels or other
trending series, test the **log returns** (or differences): a trending series is never i.i.d.,
and the test would only report the trend.

---

## 3. Using the window

Start it with `ivserial-gui` (or `python -m ivserial`, or by double-clicking). Three tabs:

* **Test** — *Open data file...*, choose the series (or tick *test all columns* to screen every
  column), choose the transform (none, differences, log returns), press *Run test*;
  *Save report...* writes a `.txt` or `.json` file. Options: permutations `B` (9999), seed, block
  length `m` and classes `d` (automatic), level, portmanteau lags and classes, patterns shown.
* **Sample size & power** — enter Gamma (and tau), or press *Use last test*; then *Required sample
  size*, *Power for n*, or *Resolution for n*.
* **Null distribution** — the limit L(m,d): mean, variance, critical values, P-value of any Q.

---

## 4. Using the command line

```
ivserial data.csv                                   # every numeric column
ivserial prices.csv -c Close -t logdiff --seed 1    # one column, log returns
ivserial examples/exchange_rates.csv -t logdiff --seed 1 --out report.txt --json results.json
ivserial plan --gamma 0.13 --tau 1.13 --m 3 --d 4 --n 350   # sample size and power
ivserial plan --v 0.093 --tau 1.13 --m 3 --d 4               # the same, from V
ivserial null --m 3 --d 4 --q 80                            # null distribution
ivserial gui                                                # the window
```

| option | meaning | default |
|---|---|---|
| `-c NAME` | column to test (repeat for several) | all numeric columns |
| `-t none/diff/logdiff` | transform before testing | `none` |
| `--B` | number of random permutations | 9999 |
| `--seed` | random seed (reproducible P-values) | none |
| `--m`, `--d` | block length and number of classes | automatic |
| `--alpha` | level of the reported decision | 0.05 |
| `--lags`, `--d2` | portmanteau lags and classes | 2, automatic |
| `--cells` | number of patterns shown | 8 |
| `--out`, `--json` | save the report / all results | — |

When several series are tested, a summary table with P-values adjusted for multiple
testing (Benjamini–Hochberg) is added.

---

## 5. Using Python

```python
import numpy as np
from ivserial import serial_test, required_sample_size

x = np.loadtxt("returns.txt")          # any 1-d array in time order
res = serial_test(x, seed=1)           # B = 9999 permutations by default
print(res)                             # full report
res.p_C, res.reject                    # combined P-value and decision
res.cramer_v, res.gamma_hat            # strength (0 to 1) and efficiency
res.tau_hat                            # standard deviation, for planning
res.cells[:5]                          # most over/under-represented patterns
res.to_dict()                          # everything, e.g. for a DataFrame

serial_test(prices, transform="logdiff", label="WTI")
```

---

## 6. Reading the output

```
Test                                 Statistic     P-value
C = min(p_Q, p_P2)  [proposed]                      0.0001
Q  (blocks, permutation)               380.369      0.0001
Q  (blocks, asymptotic)                           < 0.0001
P2 (lags, permutation)                 170.349      0.0001
P2 (lags, asymptotic)                             < 0.0001
```

* **C** is the test to use. If its P-value is below the level (e.g. 0.05), serial
  independence is rejected: the series has dependence that an i.i.d. sequence would not show.
* **Q** looks at blocks of `m` consecutive values and detects dependence that involves several
  values jointly, including dependence that is invisible in all pairs. **P2** looks at pairs
  at lags 1 and 2. C combines them exactly.
* The *permutation* P-values are exact; the *asymptotic* ones use the limit distributions
  of the paper and are accurate for n ≥ 350.
* **V-hat**, the serial Cramér coefficient, measures the strength of the dependence between 0
  and 1: 0 means that blocks of m consecutive classes are independent, 1 that the class of any one
  value determines its whole block; for m = 2 it is Cramér's V of consecutive classes. Compare values
  at the same (m, d).
* **Gamma-hat** = (d^(m−1) − 1)·V̂² is the efficiency of the test (the approximate Bahadur slope)
  and drives the sample-size calculation; it is about 0 under independence.
* **Patterns.** Each value is assigned to one of `d` classes by its rank (1 = lowest,
  `d` = highest). The table lists the sequences of `m` classes that occur most unusually
  often or rarely, with residual = (observed − expected)/√expected; |residual| > 3 is notable.
  Example (Chinese yuan): `3-3-3` occurs 41 times against 7.5 expected — runs of
  near-zero changes; `1-5-5`, `1-1-5`, `5-5-5` — clusters of large moves of either sign,
  i.e. volatility clustering.

**What a rejection does not say.** Rejection means the values are not i.i.d.; the patterns
indicate the kind of dependence, but not a particular model.

---

## 7. Residuals of a fitted model

You can apply the test to residuals of a model you have fitted (e.g. standardized GARCH
residuals) to check whether dependence remains. The P-values are then exact only
approximately. By the paper (Theorem 4.11), they are valid and slightly conservative when the
model is estimated **efficiently** (maximum likelihood with the correct innovation
distribution). After Gaussian quasi-maximum likelihood estimation with non-normal
innovations they can be misleading, especially for squared residuals; then use a
model-based bootstrap (simulate from the fitted model, re-estimate, recompute), as in
Section 4.5 of the paper.

---

## 8. Planning the sample size

```python
from ivserial import required_sample_size, approximate_power, sample_size_table
r = serial_test(pilot, seed=1)                                   # estimates Gamma and tau
required_sample_size(r.gamma_hat, r.m, r.d, power=0.8, tau=r.tau_hat)
approximate_power(0.13, n=350, m=3, d=4, tau=1.13)              # -> 0.85 (simulated: 0.82)
from ivserial import gamma_from_cramer_v                         # plan from V instead of Gamma
required_sample_size(gamma_from_cramer_v(0.1, 3, 4), 3, 4, tau=1.0)
```

This is Corollary 4.8 of the paper. With `tau = 0` the result is accurate for weak dependence
and otherwise optimistic; `serial_test` estimates `tau` from the data, so a pilot sample or a
comparable series gives a realistic plan. The report of a significant test includes this plan.

---

## 9. Details

* Resolution: `(m, d) = (3, 3)` for n < 300, `(3, ⌊(n−300)/400⌋+4)` for 300 ≤ n < 1500, and
  `m = ⌊log n / (2 log 3)⌋`, `d = 3` beyond; portmanteau classes 3, 4 or 5.
* Ties are broken at random, which keeps the test exact for discrete data.
* P-value = (1 + #{permuted statistics ≥ observed}) / (B + 1).
* Computing time: about 1 second for n = 1000 with B = 9999, and about 15 seconds for
  n = 100,000 with B = 999.
* Tests: `python tests/test_basic.py`.

## Citation

If you use the package, please cite the paper above.
Questions and bug reports: <https://github.com/emaad7/ivserial/issues>.

## License

MIT (see `LICENSE`).
