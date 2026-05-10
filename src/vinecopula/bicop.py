from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ._utils import (
    UMAX,
    UMIN,
    bisect_root,
    clip_unit,
    empirical_kendall_tau,
    legendre_integrate,
    mixed_derivative,
    n_cdf,
    n_pdf,
    n_ppf,
    partial_derivative,
    require_scipy,
)
from .families import (
    ONE_PARAMETER_FAMILIES,
    TWO_PARAMETER_FAMILIES,
    check_bicop,
    family_name,
    family_number,
    n_parameters,
    par_to_beta,
    par_to_taildep,
    par_to_tau,
    resolve_familyset,
    tau_to_par,
    tawn_a,
)


BASE_ROT_180 = {13: 3, 14: 4, 16: 6, 17: 7, 18: 8, 19: 9, 20: 10, 51: 41}
BASE_ROT_90 = {23: 3, 24: 4, 26: 6, 27: 7, 28: 8, 29: 9, 30: 10, 61: 41}
BASE_ROT_270 = {33: 3, 34: 4, 36: 6, 37: 7, 38: 8, 39: 9, 40: 10, 71: 41}
ROTATED = set(BASE_ROT_180) | set(BASE_ROT_90) | set(BASE_ROT_270)


@dataclass
class BiCop:
    family: int | str
    par: float | None = None
    par2: float = 0.0
    tau: float | None = None
    check_pars: bool = True
    familyname: str = field(init=False)
    npars: int = field(init=False)
    beta: float = field(init=False)
    taildep: dict[str, float] = field(init=False)

    def __post_init__(self):
        self.family = family_number(self.family)
        if self.tau is not None:
            self.par = float(tau_to_par(self.family, self.tau))
        if self.par is None:
            self.par = 0.0 if self.family == 0 else None
        if self.par is None:
            raise ValueError("par must be supplied unless family is independence or tau is supplied")
        self.par = float(self.par)
        self.par2 = float(self.par2)
        if self.check_pars:
            check_bicop(self.family, self.par, self.par2)
        self.familyname = family_name(self.family, short=False)
        self.npars = n_parameters(self.family)
        self.tau = float(par_to_tau(self.family, self.par, self.par2, check_pars=False))
        self.taildep = par_to_taildep(self.family, self.par, self.par2, check_pars=False)
        try:
            self.beta = float(par_to_beta(self.family, self.par, self.par2))
        except Exception:
            self.beta = float("nan")

    def pdf(self, u1, u2):
        return bicop_pdf(u1, u2, self, check_pars=False)

    def cdf(self, u1, u2):
        return bicop_cdf(u1, u2, self, check_pars=False)

    def hfunc1(self, u1, u2):
        return bicop_hfunc1(u1, u2, self, check_pars=False)

    def hfunc2(self, u1, u2):
        return bicop_hfunc2(u1, u2, self, check_pars=False)

    def sim(self, n: int, random_state=None):
        return bicop_sim(n, self, random_state=random_state)

    def __repr__(self) -> str:
        bits = [f"family={self.family} ({self.familyname})", f"par={self.par:.6g}"]
        if self.npars == 2:
            bits.append(f"par2={self.par2:.6g}")
        bits.append(f"tau={self.tau:.6g}")
        return f"BiCop({', '.join(bits)})"


def _spec(family, par=None, par2=0.0):
    if isinstance(family, BiCop):
        return family.family, family.par, family.par2
    if par is None:
        raise ValueError("par must be supplied when family is not a BiCop object")
    return family_number(family), float(par), float(par2)


def _positive_params(family: int, par: float, par2: float):
    if family in BASE_ROT_90 or family in BASE_ROT_270:
        return abs(par), abs(par2)
    return par, par2


def _tawn_cdf_base(u: float, v: float, theta: float, par2: float, par3: float) -> float:
    u = float(np.clip(u, UMIN, UMAX))
    v = float(np.clip(v, UMIN, UMAX))
    denom = math.log(u * v)
    if denom == 0:
        return min(u, v)
    w = math.log(v) / denom
    a = float(tawn_a(w, theta, par2, par3))
    return float((u * v) ** a)


def _arch_cdf_base(u: float, v: float, family: int, par: float, par2: float) -> float:
    u = float(np.clip(u, UMIN, UMAX))
    v = float(np.clip(v, UMIN, UMAX))
    if family == 0:
        return u * v
    if family == 3:
        return max((u ** (-par) + v ** (-par) - 1.0), 0.0) ** (-1.0 / par)
    if family == 4:
        return math.exp(-(((-math.log(u)) ** par + (-math.log(v)) ** par) ** (1.0 / par)))
    if family == 5:
        return -math.log1p(np.expm1(-par * u) * np.expm1(-par * v) / np.expm1(-par)) / par
    if family == 6:
        a = (1.0 - u) ** par
        b = (1.0 - v) ** par
        return 1.0 - (a + b - a * b) ** (1.0 / par)
    if family == 7:
        a = (u ** (-par) - 1.0) ** par2
        b = (v ** (-par) - 1.0) ** par2
        return (1.0 + (a + b) ** (1.0 / par2)) ** (-1.0 / par)
    if family == 8:
        a = -math.log1p(-((1.0 - u) ** par))
        b = -math.log1p(-((1.0 - v) ** par))
        return 1.0 - (1.0 - math.exp(-((a**par2 + b**par2) ** (1.0 / par2)))) ** (1.0 / par)
    if family == 9:
        a = (1.0 - (1.0 - u) ** par) ** (-par2)
        b = (1.0 - (1.0 - v) ** par) ** (-par2)
        return 1.0 - (1.0 - (a + b - 1.0) ** (-1.0 / par2)) ** (1.0 / par)
    if family == 10:
        nu = 1.0 / (1.0 - (1.0 - par2) ** par)
        a = 1.0 - (1.0 - par2 * u) ** par
        b = 1.0 - (1.0 - par2 * v) ** par
        return (1.0 - (1.0 - nu * a * b) ** (1.0 / par)) / par2
    raise NotImplementedError(f"CDF not implemented for base family {family}")


def _gaussian_cdf_scalar(u: float, v: float, rho: float) -> float:
    u = float(np.clip(u, UMIN, UMAX))
    v = float(np.clip(v, UMIN, UMAX))
    if abs(rho) < 1e-14:
        return u * v
    qv = float(n_ppf(v))
    denom = math.sqrt(max(1.0 - rho * rho, UMIN))

    def integrand(s):
        z = n_ppf(np.asarray(s))
        return n_cdf((qv - rho * z) / denom)

    return min(max(legendre_integrate(integrand, 0.0, u, n=96), 0.0), 1.0)


def _student_cdf_scalar(u: float, v: float, rho: float, df: float) -> float:
    scipy = require_scipy()
    x = scipy.stats.t.ppf(u, df)
    y = scipy.stats.t.ppf(v, df)
    return float(scipy.stats.multivariate_t.cdf([x, y], shape=[[1.0, rho], [rho, 1.0]], df=df))


def _cdf_scalar(u: float, v: float, family: int, par: float, par2: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    if u <= 0.0 or v <= 0.0:
        return 0.0
    if u >= 1.0:
        return v
    if v >= 1.0:
        return u
    if family == 1:
        return _gaussian_cdf_scalar(u, v, par)
    if family == 2:
        return _student_cdf_scalar(u, v, par, par2)
    if family in range(0, 11):
        return _arch_cdf_base(u, v, family, par, par2)
    if family in BASE_ROT_180:
        base = BASE_ROT_180[family]
        return u + v - 1.0 + _cdf_scalar(1.0 - u, 1.0 - v, base, par, par2)
    if family in BASE_ROT_90:
        base = BASE_ROT_90[family]
        pp, pp2 = abs(par), abs(par2)
        return v - _cdf_scalar(1.0 - u, v, base, pp, pp2)
    if family in BASE_ROT_270:
        base = BASE_ROT_270[family]
        pp, pp2 = abs(par), abs(par2)
        return u - _cdf_scalar(u, 1.0 - v, base, pp, pp2)
    if family == 104:
        return _tawn_cdf_base(u, v, par, par2, 1.0)
    if family == 114:
        return u + v - 1.0 + _tawn_cdf_base(1.0 - u, 1.0 - v, par, par2, 1.0)
    if family == 124:
        return v - _tawn_cdf_base(1.0 - u, v, -par, 1.0, par2)
    if family == 134:
        return u - _tawn_cdf_base(u, 1.0 - v, -par, 1.0, par2)
    if family == 204:
        return _tawn_cdf_base(u, v, par, 1.0, par2)
    if family == 214:
        return u + v - 1.0 + _tawn_cdf_base(1.0 - u, 1.0 - v, par, 1.0, par2)
    if family == 224:
        return v - _tawn_cdf_base(1.0 - u, v, -par, par2, 1.0)
    if family == 234:
        return u - _tawn_cdf_base(u, 1.0 - v, -par, par2, 1.0)
    raise NotImplementedError(f"CDF not implemented for family {family}")


def bicop_cdf(u1, u2, family, par=None, par2=0.0, *, check_pars: bool = True):
    family, par, par2 = _spec(family, par, par2)
    if check_pars:
        check_bicop(family, par, par2)
    a, b = np.broadcast_arrays(np.atleast_1d(u1).astype(float), np.atleast_1d(u2).astype(float))
    out = np.empty(a.shape, dtype=float)
    for idx in np.ndindex(a.shape):
        out[idx] = _cdf_scalar(float(a[idx]), float(b[idx]), family, par, par2)
    return out.item() if out.size == 1 else out


def _gaussian_pdf_scalar(u: float, v: float, rho: float) -> float:
    x = float(n_ppf(u))
    y = float(n_ppf(v))
    denom = max(1.0 - rho * rho, UMIN)
    return math.exp(
        (x * x + y * y) / 2.0
        + (2.0 * rho * x * y - x * x - y * y) / (2.0 * denom)
    ) / math.sqrt(denom)


def _student_pdf_scalar(u: float, v: float, rho: float, df: float) -> float:
    scipy = require_scipy()
    x = scipy.stats.t.ppf(u, df)
    y = scipy.stats.t.ppf(v, df)
    marg = scipy.stats.t.pdf(x, df) * scipy.stats.t.pdf(y, df)
    quad = (x * x + y * y - 2.0 * rho * x * y) / (df * (1.0 - rho * rho))
    joint = (
        math.gamma((df + 2.0) / 2.0)
        / (math.gamma(df / 2.0) * df * math.pi * math.sqrt(1.0 - rho * rho))
        * (1.0 + quad) ** (-(df + 2.0) / 2.0)
    )
    return float(joint / marg)


def _pdf_scalar_base(u: float, v: float, family: int, par: float, par2: float) -> float:
    u = float(np.clip(u, UMIN, UMAX))
    v = float(np.clip(v, UMIN, UMAX))
    if family == 0:
        return 1.0
    if family == 1:
        return _gaussian_pdf_scalar(u, v, par)
    if family == 2:
        return _student_pdf_scalar(u, v, par, par2)
    if family == 3:
        return max(
            (1.0 + par)
            * (u * v) ** (-1.0 - par)
            * (u ** (-par) + v ** (-par) - 1.0) ** (-2.0 - 1.0 / par),
            0.0,
        )
    if family == 4:
        t1 = (-math.log(u)) ** par + (-math.log(v)) ** par
        c = math.exp(-(t1 ** (1.0 / par)))
        return (
            c
            / (u * v)
            * t1 ** (-2.0 + 2.0 / par)
            * (math.log(u) * math.log(v)) ** (par - 1.0)
            * (1.0 + (par - 1.0) * t1 ** (-1.0 / par))
        )
    if family == 5:
        return (
            par
            * (math.exp(par) - 1.0)
            * math.exp(par * (u + v + 1.0))
            / (
                math.exp(par * (u + v))
                - math.exp(par * (v + 1.0))
                - math.exp(par * (u + 1.0))
                + math.exp(par)
            )
            ** 2
        )
    if family == 6:
        a = (1.0 - u) ** par
        b = (1.0 - v) ** par
        s = a + b - a * b
        return s ** (1.0 / par - 2.0) * (1.0 - u) ** (par - 1.0) * (1.0 - v) ** (par - 1.0) * (
            par - 1.0 + a + b - a * b
        )
    func = lambda x, y: _cdf_scalar(x, y, family, par, par2)
    return max(mixed_derivative(func, u, v), 0.0)


def _pdf_scalar(u: float, v: float, family: int, par: float, par2: float) -> float:
    if family in BASE_ROT_180:
        return _pdf_scalar(1.0 - u, 1.0 - v, BASE_ROT_180[family], par, par2)
    if family in BASE_ROT_90:
        return _pdf_scalar(1.0 - u, v, BASE_ROT_90[family], abs(par), abs(par2))
    if family in BASE_ROT_270:
        return _pdf_scalar(u, 1.0 - v, BASE_ROT_270[family], abs(par), abs(par2))
    return _pdf_scalar_base(u, v, family, par, par2)


def bicop_pdf(u1, u2, family, par=None, par2=0.0, *, check_pars: bool = True):
    family, par, par2 = _spec(family, par, par2)
    if check_pars:
        check_bicop(family, par, par2)
    a, b = np.broadcast_arrays(np.atleast_1d(u1).astype(float), np.atleast_1d(u2).astype(float))
    out = np.empty(a.shape, dtype=float)
    for idx in np.ndindex(a.shape):
        out[idx] = _pdf_scalar(float(a[idx]), float(b[idx]), family, par, par2)
    return out.item() if out.size == 1 else out


def bicop_loglik(u1, u2, family, par=None, par2=0.0, *, separate: bool = False, check_pars: bool = True):
    dens = np.asarray(bicop_pdf(u1, u2, family, par, par2, check_pars=check_pars), dtype=float)
    logs = np.log(np.maximum(dens, UMIN))
    return logs if separate else float(np.sum(logs))


def _hfunc_scalar(u1: float, u2: float, family: int, par: float, par2: float, which: int) -> float:
    u1 = float(np.clip(u1, UMIN, UMAX))
    u2 = float(np.clip(u2, UMIN, UMAX))
    if family == 0:
        return u2 if which == 1 else u1
    if family == 1:
        denom = math.sqrt(max(1.0 - par * par, UMIN))
        if which == 1:
            return float(n_cdf((float(n_ppf(u2)) - par * float(n_ppf(u1))) / denom))
        return float(n_cdf((float(n_ppf(u1)) - par * float(n_ppf(u2))) / denom))
    if family == 2:
        scipy = require_scipy()
        if which == 1:
            x = scipy.stats.t.ppf(u2, par2)
            y = scipy.stats.t.ppf(u1, par2)
        else:
            x = scipy.stats.t.ppf(u1, par2)
            y = scipy.stats.t.ppf(u2, par2)
        mu = par * y
        sigma2 = ((par2 + y * y) * (1.0 - par * par)) / (par2 + 1.0)
        return float(scipy.stats.t.cdf((x - mu) / math.sqrt(sigma2), par2 + 1.0))
    func = lambda x, y: _cdf_scalar(x, y, family, par, par2)
    axis = 0 if which == 1 else 1
    val = partial_derivative(func, u1, u2, axis=axis)
    return float(np.clip(val, 0.0, 1.0))


def bicop_hfunc1(u1, u2, family, par=None, par2=0.0, *, check_pars: bool = True):
    family, par, par2 = _spec(family, par, par2)
    if check_pars:
        check_bicop(family, par, par2)
    a, b = np.broadcast_arrays(np.atleast_1d(u1).astype(float), np.atleast_1d(u2).astype(float))
    out = np.empty(a.shape, dtype=float)
    for idx in np.ndindex(a.shape):
        out[idx] = _hfunc_scalar(float(a[idx]), float(b[idx]), family, par, par2, 1)
    return out.item() if out.size == 1 else out


def bicop_hfunc2(u1, u2, family, par=None, par2=0.0, *, check_pars: bool = True):
    family, par, par2 = _spec(family, par, par2)
    if check_pars:
        check_bicop(family, par, par2)
    a, b = np.broadcast_arrays(np.atleast_1d(u1).astype(float), np.atleast_1d(u2).astype(float))
    out = np.empty(a.shape, dtype=float)
    for idx in np.ndindex(a.shape):
        out[idx] = _hfunc_scalar(float(a[idx]), float(b[idx]), family, par, par2, 2)
    return out.item() if out.size == 1 else out


def bicop_hfunc(u1, u2, family, par=None, par2=0.0, *, check_pars: bool = True):
    return {
        "hfunc1": bicop_hfunc1(u1, u2, family, par, par2, check_pars=check_pars),
        "hfunc2": bicop_hfunc2(u1, u2, family, par, par2, check_pars=check_pars),
    }


def _hinv1_scalar(u1: float, y: float, family: int, par: float, par2: float) -> float:
    if family == 0:
        return float(np.clip(y, UMIN, UMAX))
    if family == 1:
        return float(n_cdf(par * float(n_ppf(u1)) + math.sqrt(1.0 - par * par) * float(n_ppf(y))))
    if family == 2:
        scipy = require_scipy()
        temp1 = scipy.stats.t.ppf(y, par2 + 1.0)
        temp2 = scipy.stats.t.ppf(u1, par2)
        mu = par * temp2
        var = ((par2 + temp2 * temp2) * (1.0 - par * par)) / (par2 + 1.0)
        return float(scipy.stats.t.cdf(math.sqrt(var) * temp1 + mu, par2))
    return bisect_root(lambda z: _hfunc_scalar(u1, z, family, par, par2, 1) - y)


def _hinv2_scalar(y: float, u2: float, family: int, par: float, par2: float) -> float:
    if family == 0:
        return float(np.clip(y, UMIN, UMAX))
    if family == 1:
        return float(n_cdf(par * float(n_ppf(u2)) + math.sqrt(1.0 - par * par) * float(n_ppf(y))))
    if family == 2:
        scipy = require_scipy()
        temp1 = scipy.stats.t.ppf(y, par2 + 1.0)
        temp2 = scipy.stats.t.ppf(u2, par2)
        mu = par * temp2
        var = ((par2 + temp2 * temp2) * (1.0 - par * par)) / (par2 + 1.0)
        return float(scipy.stats.t.cdf(math.sqrt(var) * temp1 + mu, par2))
    return bisect_root(lambda z: _hfunc_scalar(z, u2, family, par, par2, 2) - y)


def bicop_hinv1(u1, u2, family, par=None, par2=0.0, *, check_pars: bool = True):
    family, par, par2 = _spec(family, par, par2)
    if check_pars:
        check_bicop(family, par, par2)
    a, b = np.broadcast_arrays(np.atleast_1d(u1).astype(float), np.atleast_1d(u2).astype(float))
    out = np.empty(a.shape, dtype=float)
    for idx in np.ndindex(a.shape):
        out[idx] = _hinv1_scalar(float(a[idx]), float(b[idx]), family, par, par2)
    return out.item() if out.size == 1 else out


def bicop_hinv2(u1, u2, family, par=None, par2=0.0, *, check_pars: bool = True):
    family, par, par2 = _spec(family, par, par2)
    if check_pars:
        check_bicop(family, par, par2)
    a, b = np.broadcast_arrays(np.atleast_1d(u1).astype(float), np.atleast_1d(u2).astype(float))
    out = np.empty(a.shape, dtype=float)
    for idx in np.ndindex(a.shape):
        out[idx] = _hinv2_scalar(float(a[idx]), float(b[idx]), family, par, par2)
    return out.item() if out.size == 1 else out


def bicop_hinv(u1, u2, family, par=None, par2=0.0, *, check_pars: bool = True):
    return {
        "hinv1": bicop_hinv1(u1, u2, family, par, par2, check_pars=check_pars),
        "hinv2": bicop_hinv2(u1, u2, family, par, par2, check_pars=check_pars),
    }


def bicop_sim(n: int, family, par=None, par2=0.0, *, random_state=None):
    family, par, par2 = _spec(family, par, par2)
    check_bicop(family, par, par2)
    rng = np.random.default_rng(random_state)
    w = rng.uniform(size=(int(n), 2))
    out2 = bicop_hinv1(w[:, 0], w[:, 1], family, par, par2, check_pars=False)
    return np.column_stack([w[:, 0], out2])


def bicop_condsim(n: int, cond, family, par=None, par2=0.0, *, cond_var: int = 1, random_state=None):
    family, par, par2 = _spec(family, par, par2)
    rng = np.random.default_rng(random_state)
    cond = np.broadcast_to(np.asarray(cond, dtype=float), (int(n),))
    y = rng.uniform(size=int(n))
    if cond_var == 1:
        return bicop_hinv1(cond, y, family, par, par2, check_pars=False)
    if cond_var == 2:
        return bicop_hinv2(y, cond, family, par, par2, check_pars=False)
    raise ValueError("cond_var must be 1 or 2")


def bicop_est(u1, u2, family=1, *, method: str = "itau", par2: float | None = None, se: bool = False):
    family = family_number(family)
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    mask = np.isfinite(u1) & np.isfinite(u2)
    u1 = u1[mask]
    u2 = u2[mask]
    tau = empirical_kendall_tau(u1, u2)
    if method.lower() in {"itau", "tau"} or family in ONE_PARAMETER_FAMILIES or family == 2:
        par = float(tau_to_par(family, tau, check_taus=True))
        p2 = 4.0 if family == 2 and par2 is None else (0.0 if par2 is None else float(par2))
        fit = BiCop(family, par, p2, check_pars=True)
    else:
        scipy = require_scipy()
        start = np.array([float(tau_to_par(3, max(min(abs(tau), 0.9), 1e-3), check_taus=False)), 1.5])

        def objective(x):
            try:
                return -bicop_loglik(u1, u2, family, float(x[0]), float(x[1]), check_pars=True)
            except Exception:
                return 1e100

        res = scipy.optimize.minimize(objective, start, method="Nelder-Mead")
        fit = BiCop(family, float(res.x[0]), float(res.x[1]), check_pars=False)
    ll = bicop_loglik(u1, u2, fit, check_pars=False)
    fit.nobs = int(len(u1))  # type: ignore[attr-defined]
    fit.logLik = ll  # type: ignore[attr-defined]
    fit.AIC = -2.0 * ll + 2.0 * fit.npars  # type: ignore[attr-defined]
    fit.BIC = -2.0 * ll + math.log(max(len(u1), 1)) * fit.npars  # type: ignore[attr-defined]
    fit.emptau = tau  # type: ignore[attr-defined]
    return fit


def bicop_select(u1, u2, familyset: Iterable[int] | str | None = "all", *, selectioncrit: str = "AIC"):
    familyset = resolve_familyset(familyset)
    fits = []
    for fam in familyset:
        try:
            fit = bicop_est(u1, u2, fam, method="itau")
            fits.append(fit)
        except Exception:
            continue
    if not fits:
        raise ValueError("No candidate family could be estimated")
    key = "BIC" if selectioncrit.upper() == "BIC" else "AIC"
    return min(fits, key=lambda f: getattr(f, key))


def bicop_aic(u1, u2, family, par=None, par2=0.0):
    family, par, par2 = _spec(family, par, par2)
    npar = n_parameters(family)
    return -2.0 * bicop_loglik(u1, u2, family, par, par2, check_pars=False) + 2.0 * npar


def bicop_bic(u1, u2, family, par=None, par2=0.0):
    family, par, par2 = _spec(family, par, par2)
    n = len(np.atleast_1d(u1))
    npar = n_parameters(family)
    return -2.0 * bicop_loglik(u1, u2, family, par, par2, check_pars=False) + math.log(max(n, 1)) * npar


def bicop_compare(u1, u2, obj1, obj2):
    ll1 = bicop_loglik(u1, u2, obj1, separate=True, check_pars=False)
    ll2 = bicop_loglik(u1, u2, obj2, separate=True, check_pars=False)
    diff = np.asarray(ll1) - np.asarray(ll2)
    return {
        "logLik1": float(np.sum(ll1)),
        "logLik2": float(np.sum(ll2)),
        "mean_diff": float(np.mean(diff)),
        "statistic": float(np.mean(diff) / (np.std(diff, ddof=1) / math.sqrt(len(diff)))) if len(diff) > 1 else float("nan"),
    }


def bicop_lambda(u1, u2):
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    q = np.linspace(0.05, 0.95, 19)
    lower = np.array([np.mean((u1 <= qq) & (u2 <= qq)) / qq for qq in q])
    upper = np.array([np.mean((u1 > qq) & (u2 > qq)) / (1.0 - qq) for qq in q])
    return {"q": q, "lower": lower, "upper": upper}


def bicop_ind_test(u1, u2):
    tau = empirical_kendall_tau(u1, u2)
    n = len(np.asarray(u1))
    # Large-sample Kendall tau normal approximation.
    var = 2.0 * (2.0 * n + 5.0) / (9.0 * n * (n - 1.0)) if n > 1 else float("nan")
    stat = tau / math.sqrt(var) if var > 0 else float("nan")
    p = 2.0 * (1.0 - float(n_cdf(abs(stat)))) if math.isfinite(stat) else float("nan")
    return {"statistic": stat, "p_value": p, "tau": tau}


BiCopPDF = bicop_pdf
BiCopCDF = bicop_cdf
BiCopHfunc = bicop_hfunc
BiCopHfunc1 = bicop_hfunc1
BiCopHfunc2 = bicop_hfunc2
BiCopHinv = bicop_hinv
BiCopHinv1 = bicop_hinv1
BiCopHinv2 = bicop_hinv2
BiCopSim = bicop_sim
BiCopCondSim = bicop_condsim
BiCopLogLik = bicop_loglik
BiCopEst = bicop_est
BiCopSelect = bicop_select
BiCopAIC = bicop_aic
BiCopBIC = bicop_bic
BiCopCompare = bicop_compare
BiCopVuongClarke = bicop_compare
BiCopLambda = bicop_lambda
BiCopIndTest = bicop_ind_test
