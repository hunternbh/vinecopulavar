from __future__ import annotations

import math
from statistics import NormalDist
from typing import Callable, Iterable, Sequence

import numpy as np

EPS = 1e-12
UMIN = 1e-12
UMAX = 1.0 - 1e-12
_NORMAL = NormalDist()


def clip_unit(x):
    return np.clip(np.asarray(x, dtype=float), UMIN, UMAX)


def as_1d(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float))


def scalarize(x):
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def broadcast_arrays(*values):
    arrays = [np.atleast_1d(np.asarray(v, dtype=float)) for v in values]
    return np.broadcast_arrays(*arrays)


def n_cdf(x):
    arr = np.asarray(x, dtype=float)
    vec = np.vectorize(_NORMAL.cdf, otypes=[float])
    return vec(arr)


def n_ppf(p):
    arr = clip_unit(p)
    vec = np.vectorize(_NORMAL.inv_cdf, otypes=[float])
    return vec(arr)


def n_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def require_scipy():
    try:
        import scipy  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "This operation requires SciPy. Install with "
            "`pip install vinecopula-native[scipy]`."
        ) from exc
    return scipy


def legendre_integrate(
    func: Callable[[np.ndarray], np.ndarray | float],
    a: float,
    b: float,
    *,
    n: int = 96,
) -> float:
    if a == b:
        return 0.0
    if b < a:
        return -legendre_integrate(func, b, a, n=n)
    x, w = np.polynomial.legendre.leggauss(n)
    pts = 0.5 * (b - a) * x + 0.5 * (a + b)
    vals = np.asarray(func(pts), dtype=float)
    return float(0.5 * (b - a) * np.sum(w * vals))


def bisect_root(
    func: Callable[[float], float],
    lo: float = UMIN,
    hi: float = UMAX,
    *,
    tol: float = 1e-10,
    maxiter: int = 100,
) -> float:
    flo = func(lo)
    fhi = func(hi)
    if abs(flo) <= tol:
        return lo
    if abs(fhi) <= tol:
        return hi
    if flo * fhi > 0:
        # Conditional distributions should bracket on the unit interval; if
        # roundoff breaks that property, return the closer endpoint.
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fmid = func(mid)
        if abs(fmid) <= tol or (hi - lo) <= tol:
            return mid
        if flo * fmid <= 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)


def partial_derivative(
    func: Callable[[float, float], float],
    u: float,
    v: float,
    *,
    axis: int,
    step: float = 1e-5,
) -> float:
    if axis == 0:
        lo = max(UMIN, u - step)
        hi = min(UMAX, u + step)
        if hi == lo:
            return 0.0
        return (func(hi, v) - func(lo, v)) / (hi - lo)
    lo = max(UMIN, v - step)
    hi = min(UMAX, v + step)
    if hi == lo:
        return 0.0
    return (func(u, hi) - func(u, lo)) / (hi - lo)


def mixed_derivative(
    func: Callable[[float, float], float],
    u: float,
    v: float,
    *,
    step: float = 2e-5,
) -> float:
    u0 = max(UMIN, u - step)
    u1 = min(UMAX, u + step)
    v0 = max(UMIN, v - step)
    v1 = min(UMAX, v + step)
    denom = (u1 - u0) * (v1 - v0)
    if denom == 0:
        return 0.0
    return (func(u1, v1) - func(u1, v0) - func(u0, v1) + func(u0, v0)) / denom


def digamma(x: float) -> float:
    result = 0.0
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    return result + math.log(x) - 0.5 * inv - inv2 * (
        1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0)
    )


def trigamma(x: float) -> float:
    result = 0.0
    while x < 8.0:
        result += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    return result + inv + 0.5 * inv2 + inv2 * inv / 6.0 - inv2 * inv2 * inv / 30.0


def _count_inversions(values: np.ndarray) -> int:
    values = np.asarray(values)
    if len(values) < 2:
        return 0
    tmp = np.empty_like(values)

    def sort_count(lo: int, hi: int) -> int:
        if hi - lo <= 1:
            return 0
        mid = (lo + hi) // 2
        inv = sort_count(lo, mid) + sort_count(mid, hi)
        i, j, k = lo, mid, lo
        while i < mid and j < hi:
            if values[i] <= values[j]:
                tmp[k] = values[i]
                i += 1
            else:
                tmp[k] = values[j]
                j += 1
                inv += mid - i
            k += 1
        while i < mid:
            tmp[k] = values[i]
            i += 1
            k += 1
        while j < hi:
            tmp[k] = values[j]
            j += 1
            k += 1
        values[lo:hi] = tmp[lo:hi]
        return inv

    return sort_count(0, len(values))


def _tie_pairs(values: np.ndarray) -> int:
    if len(values) < 2:
        return 0
    _, counts = np.unique(values, return_counts=True, axis=0 if values.ndim > 1 else None)
    return int(np.sum(counts * (counts - 1) // 2))


def empirical_kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 2:
        return float("nan")
    order = np.lexsort((y, x))
    x_sorted = x[order]
    y_sorted = y[order]
    n0 = n * (n - 1) // 2
    n1 = _tie_pairs(x_sorted)
    n2 = _tie_pairs(y_sorted)
    xy = np.column_stack([x_sorted, y_sorted])
    n3 = _tie_pairs(xy)
    discordant = _count_inversions(y_sorted.copy())
    numerator = n0 - n1 - n2 + n3 - 2 * discordant
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    return float(numerator / denom) if denom > 0 else float("nan")


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def ensure_square_matrix(x, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return arr
