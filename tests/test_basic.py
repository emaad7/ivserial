"""Basic checks: exact size under independence, power under dependence, API."""
import numpy as np
from ivserial import serial_test, required_sample_size, choose_resolution, null_limit_sf


def test_api_and_reproducibility():
    x = np.random.default_rng(0).standard_normal(400)
    a, b = serial_test(x, B=499, seed=3), serial_test(x, B=499, seed=3)
    assert a.p_C == b.p_C and 0 < a.p_C <= 1 and (a.m, a.d) == choose_resolution(400)


def test_size_under_independence():
    rng = np.random.default_rng(1)
    rej = [serial_test(rng.standard_t(3, 200), B=199, seed=i).p_C <= 0.05 for i in range(200)]
    assert 0.01 <= np.mean(rej) <= 0.10


def test_power_against_arch():
    rng = np.random.default_rng(2)
    n = 1000; e = rng.standard_normal(n + 200); z = np.zeros(n + 200)
    for t in range(1, n + 200):
        z[t] = np.sqrt(1 + 0.8 * z[t - 1] ** 2) * e[t]
    r = serial_test(z[200:], B=499, seed=1)
    assert r.reject and r.gamma_hat > 0.05


def test_limit_and_planning():
    assert abs(null_limit_sf(33.76, 3, 3) - 0.05) < 0.002
    assert required_sample_size(0.05, m=3, d=4) > required_sample_size(0.2, m=3, d=4)


if __name__ == "__main__":
    for f in [test_api_and_reproducibility, test_size_under_independence, test_power_against_arch,
              test_limit_and_planning]:
        f(); print("ok", f.__name__)
