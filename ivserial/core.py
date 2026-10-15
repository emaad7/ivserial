"""Exact symbolic tests of serial independence.

Implements the procedure of Ashtari Nezhad, "Information Variance and Exact
Symbolic Tests of Serial Independence":

* the block statistic Q (Pearson's statistic for the frequencies of blocks of
  m consecutive quantile classes, d classes),
* the symbolic portmanteau P2 (lag-1 and lag-2 Pearson statistics of pairs of
  classes),
* the combined test C = min(p_Q, p_P2), calibrated exactly from the same
  random permutations,
* the asymptotic P-value of Q from its explicit weighted chi-square limit,
* the bias-corrected normalized information variance Gamma-hat (strength of
  dependence on the scale of the Bahadur slope), and
* the standardized cell residuals, which show which patterns of consecutive
  values are over- or under-represented.

All tests are rank based: under the null hypothesis of independent and
identically distributed observations their permutation P-values are exact
for every sample size and every continuous distribution.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy import integrate, stats

__all__ = ["serial_test", "SerialTestResult", "choose_resolution", "null_limit_sf", "null_limit_quantile",
           "null_limit_summary", "required_sample_size", "approximate_power", "sample_size_table",
           "serial_cramer_v", "gamma_from_cramer_v"]


# --------------------------------------------------------------------------
# Resolution rules (Section 6.7 of the paper)
# --------------------------------------------------------------------------
def choose_resolution(n: int):
    """Block length m and number of classes d for the block statistic Q."""
    if n < 300:
        return 3, 3
    if n < 1500:
        return 3, (n - 300) // 400 + 4
    return int(math.floor(math.log(n) / (2 * math.log(3)))), 3


def _portmanteau_classes(n: int) -> int:
    return 3 if n < 300 else (4 if n < 800 else 5)


# --------------------------------------------------------------------------
# Null limit of Q (Theorem 4.1): L_{m,d} = sum_{s=2}^m (m-s+1) chi2_{(d-1)^2 d^{s-2}}
# --------------------------------------------------------------------------
def _limit_weights(m: int, d: int):
    w = np.array([m - s + 1 for s in range(2, m + 1)], float)
    df = np.array([(d - 1) ** 2 * d ** (s - 2) for s in range(2, m + 1)], float)
    return w, df


def null_limit_moments(m: int, d: int):
    w, df = _limit_weights(m, d)
    return float(np.sum(w * df)), float(2 * np.sum(w ** 2 * df))


def null_limit_sf(t: float, m: int, d: int) -> float:
    """P(L_{m,d} > t), by numerical inversion of the characteristic function (Imhof, 1961)."""
    w, df = _limit_weights(m, d)
    if len(w) == 1:
        return float(stats.chi2.sf(t / w[0], df[0]))
    mean, var = float(np.sum(w * df)), float(2 * np.sum(w ** 2 * df))
    if t > mean + 12.0 * math.sqrt(var):
        th = 0.25 / w.max()
        return float(min(1.0, math.exp(-th * t - 0.5 * np.sum(df * np.log(1 - 2 * th * w)))))

    def integrand(u):
        theta = 0.5 * np.sum(df * np.arctan(w * u)) - 0.5 * t * u
        rho = np.prod((1.0 + (w * u) ** 2) ** (df / 4.0))
        return np.sin(theta) / (u * rho)

    val, _ = integrate.quad(integrand, 0.0, np.inf, limit=400)
    return float(min(max(0.5 + val / math.pi, 0.0), 1.0))


def null_limit_quantile(alpha: float, m: int, d: int) -> float:
    from scipy.optimize import brentq
    mean, var = null_limit_moments(m, d)
    hi = mean + 20 * math.sqrt(var)
    return float(brentq(lambda t: null_limit_sf(t, m, d) - alpha, 1e-9, hi))


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------
def _block_codes(delta: np.ndarray, m: int, d: int) -> np.ndarray:
    K = delta.shape[-1] - m + 1
    c = np.zeros(delta.shape[:-1] + (K,), dtype=np.int64)
    for j in range(m):
        c = c * d + delta[..., j:j + K]
    return c


def _counts(codes: np.ndarray, R: int) -> np.ndarray:
    codes = np.atleast_2d(codes)
    B = codes.shape[0]
    off = (np.arange(B, dtype=np.int64) * R)[:, None]
    return np.bincount((codes + off).ravel(), minlength=B * R).reshape(B, R)


def _pearson_blocks(delta: np.ndarray, m: int, d: int) -> np.ndarray:
    R = d ** m
    K = delta.shape[-1] - m + 1
    P = _counts(_block_codes(delta, m, d), R) / K
    return K * R * np.sum((P - 1.0 / R) ** 2, axis=1)


def _portmanteau(delta: np.ndarray, d: int, lags: int) -> np.ndarray:
    delta = np.atleast_2d(delta)
    B, n = delta.shape
    off = (np.arange(B, dtype=np.int64) * d * d)[:, None]
    out = np.zeros(B)
    for j in range(1, lags + 1):
        c = delta[:, :-j] * d + delta[:, j:]
        cnt = np.bincount((c + off).ravel(), minlength=B * d * d).reshape(B, d * d)
        N = n - j
        out += N * d * d * np.sum((cnt / N - 1.0 / d ** 2) ** 2, axis=1)
    return out


def _ranks(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """0-based ranks; ties are broken at random, which keeps the test exact."""
    order = np.lexsort((rng.random(x.size), x))
    r = np.empty(x.size, dtype=np.int64)
    r[order] = np.arange(x.size)
    return r


# --------------------------------------------------------------------------
# Result object
# --------------------------------------------------------------------------
@dataclass
class Cell:
    pattern: str
    observed: int
    expected: float
    residual: float


@dataclass
class SerialTestResult:
    n: int
    m: int
    d: int
    d_portmanteau: int
    lags: int
    B: int
    Q: float
    p_Q: float
    p_Q_asymptotic: float
    P2: float
    p_P2: float
    p_P2_asymptotic: float
    p_C: float
    gamma_hat: float
    alpha: float
    tau_hat: float = 0.0
    cramer_v: float = 0.0
    cells: List[Cell] = field(default_factory=list)
    n_removed: int = 0
    label: str = ""

    @property
    def reject(self) -> bool:
        """Decision of the combined test C at level alpha."""
        return self.p_C <= self.alpha

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("label", "n", "m", "d", "d_portmanteau", "lags", "B", "Q", "p_Q",
                                           "p_Q_asymptotic", "P2", "p_P2", "p_P2_asymptotic", "p_C",
                                           "gamma_hat", "cramer_v", "tau_hat", "alpha", "n_removed")}
        d["reject"] = self.reject
        d["cells"] = [c.__dict__ for c in self.cells]
        return d

    def summary(self, n_cells: int = 8) -> str:
        L = []
        title = "Exact symbolic test of serial independence"
        if self.label:
            title += f"  [{self.label}]"
        L.append(title)
        L.append("=" * len(title))
        L.append(f"Observations n = {self.n}" + (f"  ({self.n_removed} missing values removed)" if self.n_removed else ""))
        L.append(f"Block statistic: m = {self.m} consecutive values, d = {self.d} quantile classes "
                 f"({self.d ** self.m} patterns)")
        L.append(f"Portmanteau: lags 1..{self.lags}, d = {self.d_portmanteau} classes")
        L.append(f"Random permutations B = {self.B}")
        L.append("")
        L.append(f"{'Test':<34}{'Statistic':>12}{'P-value':>12}")
        L.append(f"{'C = min(p_Q, p_P2)  [proposed]':<34}{'':>12}{_fmt_p(self.p_C):>12}")
        L.append(f"{'Q  (blocks, permutation)':<34}{self.Q:>12.3f}{_fmt_p(self.p_Q):>12}")
        L.append(f"{'Q  (blocks, asymptotic)':<34}{'':>12}{_fmt_p(self.p_Q_asymptotic):>12}")
        L.append(f"{'P2 (lags, permutation)':<34}{self.P2:>12.3f}{_fmt_p(self.p_P2):>12}")
        L.append(f"{'P2 (lags, asymptotic)':<34}{'':>12}{_fmt_p(self.p_P2_asymptotic):>12}")
        L.append("")
        verdict = "REJECTED" if self.reject else "not rejected"
        L.append(f"Serial independence is {verdict} at the {self.alpha:g} level (combined test C).")
        L.append(f"Strength of dependence: V-hat = {self.cramer_v:.3f}")
        L.append("  (serial Cramer coefficient, 0 to 1: 0 = independent blocks, 1 = each value determines its block)")
        L.append(f"Efficiency: Gamma-hat = {self.gamma_hat:.4f}")
        L.append("  (bias-corrected normalized information variance = approximate Bahadur slope; about 0 under independence)")
        if self.reject and self.gamma_hat > 0:
            try:
                n80 = required_sample_size(self.gamma_hat, self.m, self.d, 0.05, 0.8, self.tau_hat)
                L.append(f"Planning: power 0.8 against dependence of this strength needs about {n80} observations")
                L.append(f"  (tau-hat = {self.tau_hat:.3f}; Corollary 4.8 of the paper).")
            except Exception:  # noqa: BLE001
                pass
        if self.cells:
            L.append("")
            L.append(f"Most over- and under-represented patterns (classes 1 = lowest ... {self.d} = highest):")
            L.append(f"{'pattern':<14}{'observed':>10}{'expected':>10}{'residual':>10}")
            for c in self.cells[:n_cells]:
                L.append(f"{c.pattern:<14}{c.observed:>10d}{c.expected:>10.1f}{c.residual:>10.2f}")
            L.append("Residual = (observed - expected)/sqrt(expected); |residual| > 3 is notable.")
        return "\n".join(L)

    __str__ = summary


def _tau_hat(codes: np.ndarray, p: np.ndarray, R: int, m: int) -> float:
    """Estimate of tau in Theorem 4.5(iv) for Pearson's statistic: the long-run standard
    deviation of g_t = 2R p_hat(c_t), the gradient of Gamma_P applied to the block indicators
    (Bartlett kernel, bandwidth m - 1 + floor(4 (K/100)^(2/9)))."""
    g = 2.0 * R * p[codes]
    g = g - g.mean()
    K = g.size
    L = m - 1 + int(math.floor(4 * (K / 100.0) ** (2.0 / 9.0)))
    v = float(np.dot(g, g) / K)
    for h in range(1, min(L, K - 1) + 1):
        v += 2.0 * (1.0 - h / (L + 1.0)) * float(np.dot(g[:-h], g[h:]) / K)
    return math.sqrt(max(v, 0.0))


def serial_cramer_v(gamma: float, m: int, d: int) -> float:
    """Serial Cramer coefficient V = sqrt(max(Gamma, 0) / (d^(m-1) - 1)), between 0 and 1
    (Proposition 4.7 of the paper); for m = 2 it is Cramer's V of consecutive classes."""
    return float(math.sqrt(max(gamma, 0.0) / (d ** (m - 1) - 1)))


def gamma_from_cramer_v(v: float, m: int, d: int) -> float:
    """Inverse of serial_cramer_v: Gamma = (d^(m-1) - 1) V^2."""
    if not 0 <= v <= 1:
        raise ValueError("V must be between 0 and 1")
    return float((d ** (m - 1) - 1) * v * v)


def _fmt_p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


# --------------------------------------------------------------------------
# Main function
# --------------------------------------------------------------------------
def serial_test(x: Sequence[float], m: Optional[int] = None, d: Optional[int] = None,
                d_portmanteau: Optional[int] = None, lags: int = 2, B: int = 9999,
                seed: Optional[int] = None, alpha: float = 0.05, transform: Optional[str] = None,
                label: str = "", n_cells: int = 10, chunk: int = 500) -> SerialTestResult:
    """Exact symbolic test of serial independence.

    Parameters
    ----------
    x : sequence of numbers
        The time series, in time order. Missing values (NaN) are removed.
    m, d : int, optional
        Block length and number of quantile classes of the block statistic Q.
        By default chosen from n by the rule of the paper.
    d_portmanteau : int, optional
        Number of classes of the portmanteau P2 (default: 3, 4 or 5 by n).
    lags : int
        Lags of the portmanteau (default 2).
    B : int
        Number of random permutations (default 9999). The P-values are exact
        for every B; larger B makes them less variable.
    seed : int, optional
        Seed of the random number generator, for reproducible P-values.
    alpha : float
        Level used for the reported decision.
    transform : {None, "diff", "logdiff"}
        Apply first differences or log differences (log returns, in percent)
        before testing; use "logdiff" for prices.
    """
    x = np.asarray(x, dtype=float).ravel()
    if transform in ("diff", "logdiff"):
        if transform == "logdiff":
            if np.any(x[np.isfinite(x)] <= 0):
                raise ValueError("log differences require positive values")
            x = 100.0 * np.diff(np.log(x))
        else:
            x = np.diff(x)
    elif transform not in (None, "none", ""):
        raise ValueError("transform must be None, 'diff' or 'logdiff'")
    ok = np.isfinite(x)
    n_removed = int(np.sum(~ok))
    if n_removed:
        warnings.warn(f"{n_removed} missing or non-finite values removed")
    x = x[ok]
    n = x.size
    if n < 20:
        raise ValueError(f"at least 20 observations are needed (got {n})")
    if np.unique(x).size < 3:
        raise ValueError("the series is (almost) constant")

    if m is None or d is None:
        m0, d0 = choose_resolution(n)
        m = m0 if m is None else int(m)
        d = d0 if d is None else int(d)
    d2 = _portmanteau_classes(n) if d_portmanteau is None else int(d_portmanteau)
    if m < 2 or d < 2:
        raise ValueError("m and d must be at least 2")
    R, K = d ** m, n - m + 1
    if K / R < 1:
        warnings.warn(f"only {K / R:.2f} blocks per pattern on average; consider a smaller m or d")

    rng = np.random.default_rng(seed)
    r = _ranks(x, rng)
    dq, dp = (d * r) // n, (d2 * r) // n
    Q0 = float(_pearson_blocks(dq[None, :], m, d)[0])
    P0 = float(_portmanteau(dp[None, :], d2, lags)[0])

    Qs, Ps = [], []
    done = 0
    while done < B:
        b = min(chunk, B - done)
        perm = rng.permuted(np.broadcast_to(r, (b, n)).copy(), axis=1)
        Qs.append(_pearson_blocks((d * perm) // n, m, d))
        Ps.append(_portmanteau((d2 * perm) // n, d2, lags))
        done += b
    Qs, Ps = np.concatenate(Qs), np.concatenate(Ps)

    def upper_p(v, v0):
        return float((1.0 + np.sum(v >= v0 - 1e-9 * abs(v0))) / (B + 1.0))

    pQ, pP = upper_p(Qs, Q0), upper_p(Ps, P0)
    # exact minimum-P combination within the same permutations
    allQ, allP = np.concatenate([[Q0], Qs]), np.concatenate([[P0], Ps])

    def rank_p(v):
        s = np.sort(v)
        return (v.size - np.searchsorted(s, v - 1e-9 * np.abs(v), side="left")) / v.size

    pm = np.minimum(rank_p(allQ), rank_p(allP))
    pC = float((1.0 + np.sum(pm[1:] <= pm[0] + 1e-12)) / (B + 1.0))

    mu0, _ = null_limit_moments(m, d)
    gamma_hat = (Q0 - mu0) / K
    pQa = null_limit_sf(Q0, m, d)
    pPa = float(stats.chi2.sf(P0, lags * (d2 - 1) ** 2))

    codes_obs = _block_codes(dq, m, d)
    counts = np.bincount(codes_obs, minlength=R)
    tau_hat = _tau_hat(codes_obs, counts / K, R, m)
    e = K / R
    res = (counts - e) / math.sqrt(e)
    order = np.argsort(-np.abs(res))[:n_cells]
    cells = []
    for c in order:
        digits, v = [], int(c)
        for _ in range(m):
            digits.append(v % d + 1)
            v //= d
        cells.append(Cell("-".join(str(t) for t in reversed(digits)), int(counts[c]), e, float(res[c])))

    return SerialTestResult(n=n, m=m, d=d, d_portmanteau=d2, lags=lags, B=B, Q=Q0, p_Q=pQ,
                            p_Q_asymptotic=pQa, P2=P0, p_P2=pP, p_P2_asymptotic=pPa, p_C=pC,
                            gamma_hat=float(gamma_hat), tau_hat=float(tau_hat),
                            cramer_v=serial_cramer_v(gamma_hat, m, d), alpha=alpha, cells=cells,
                            n_removed=n_removed, label=label)


# --------------------------------------------------------------------------
# Sample-size planning (Corollary 4.8)
# --------------------------------------------------------------------------
def required_sample_size(gamma: float, m: int = 3, d: int = 4, alpha: float = 0.05,
                         power: float = 0.8, tau: float = 0.0) -> int:
    """Approximate number of observations for the block test to reach the given power
    against dependence of strength gamma (normalized information variance).

    Uses power ~ Phi((K*gamma + mu0 - q_alpha) / sqrt(K*tau^2 + V0)); with tau = 0
    (weak dependence) K = (q_alpha - mu0 + z_power*sqrt(V0)) / gamma.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    mu0, V0 = null_limit_moments(m, d)
    q = null_limit_quantile(alpha, m, d)
    z = stats.norm.ppf(power)
    if tau <= 0:
        K = (q - mu0 + z * math.sqrt(V0)) / gamma
    else:
        # solve K*gamma + mu0 - q = z*sqrt(K*tau^2 + V0) for K (quadratic in sqrt form)
        a = gamma ** 2
        bq = -2 * gamma * (q - mu0) - z * z * tau ** 2
        c = (q - mu0) ** 2 - z * z * V0
        K = (-bq + math.sqrt(max(bq * bq - 4 * a * c, 0.0))) / (2 * a)
    return int(math.ceil(K)) + m - 1


def approximate_power(gamma: float, n: int, m: int = 3, d: int = 4, alpha: float = 0.05,
                      tau: float = 0.0) -> float:
    """Approximate power of the block test with n observations against dependence of
    strength gamma (Corollary 4.8): Phi((K*gamma + mu0 - q_alpha) / sqrt(K*tau^2 + V0))."""
    K = n - m + 1
    mu0, V0 = null_limit_moments(m, d)
    q = null_limit_quantile(alpha, m, d)
    return float(stats.norm.cdf((K * gamma + mu0 - q) / math.sqrt(K * tau ** 2 + V0)))


def sample_size_table(gamma: float, m: int = 3, d: int = 4, alpha: float = 0.05, tau: float = 0.0,
                      powers=(0.5, 0.8, 0.9, 0.95)):
    """List of (power, required number of observations)."""
    return [(p, required_sample_size(gamma, m, d, alpha, p, tau)) for p in powers]


def null_limit_summary(m: int, d: int) -> dict:
    """Mean, variance, components and upper critical values of the null limit L_{m,d}."""
    w, df = _limit_weights(m, d)
    mean, var = null_limit_moments(m, d)
    return {"m": m, "d": d, "patterns": d ** m,
            "components": [(float(a), int(b)) for a, b in zip(w, df)],
            "mean": mean, "variance": var,
            "critical": {a: null_limit_quantile(a, m, d) for a in (0.10, 0.05, 0.01)}}
