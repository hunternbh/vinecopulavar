from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._utils import UMAX, UMIN, bisect_root, digamma, legendre_integrate, require_scipy, trigamma


@dataclass(frozen=True)
class FamilyInfo:
    code: int
    short: str
    long: str


FAMILY_INFO: dict[int, FamilyInfo] = {
    0: FamilyInfo(0, "I", "Independence"),
    1: FamilyInfo(1, "N", "Gaussian"),
    2: FamilyInfo(2, "t", "t"),
    3: FamilyInfo(3, "C", "Clayton"),
    4: FamilyInfo(4, "G", "Gumbel"),
    5: FamilyInfo(5, "F", "Frank"),
    6: FamilyInfo(6, "J", "Joe"),
    7: FamilyInfo(7, "BB1", "BB1"),
    8: FamilyInfo(8, "BB6", "BB6"),
    9: FamilyInfo(9, "BB7", "BB7"),
    10: FamilyInfo(10, "BB8", "BB8"),
    13: FamilyInfo(13, "SC", "Survival Clayton"),
    14: FamilyInfo(14, "SG", "Survival Gumbel"),
    16: FamilyInfo(16, "SJ", "Survival Joe"),
    17: FamilyInfo(17, "SBB1", "Survival BB1"),
    18: FamilyInfo(18, "SBB6", "Survival BB6"),
    19: FamilyInfo(19, "SBB7", "Survival BB7"),
    20: FamilyInfo(20, "SBB8", "Survival BB8"),
    23: FamilyInfo(23, "C90", "Rotated Clayton 90 degrees"),
    24: FamilyInfo(24, "G90", "Rotated Gumbel 90 degrees"),
    26: FamilyInfo(26, "J90", "Rotated Joe 90 degrees"),
    27: FamilyInfo(27, "BB1_90", "Rotated BB1 90 degrees"),
    28: FamilyInfo(28, "BB6_90", "Rotated BB6 90 degrees"),
    29: FamilyInfo(29, "BB7_90", "Rotated BB7 90 degrees"),
    30: FamilyInfo(30, "BB8_90", "Rotated Frank-Joe 90 degrees"),
    33: FamilyInfo(33, "C270", "Rotated Clayton 270 degrees"),
    34: FamilyInfo(34, "G270", "Rotated Gumbel 270 degrees"),
    36: FamilyInfo(36, "J270", "Rotated Joe 270 degrees"),
    37: FamilyInfo(37, "BB1_270", "Rotated BB1 270 degrees"),
    38: FamilyInfo(38, "BB6_270", "Rotated BB6 270 degrees"),
    39: FamilyInfo(39, "BB7_270", "Rotated BB7 270 degrees"),
    40: FamilyInfo(40, "BB8_270", "Rotated Frank-Joe 270 degrees"),
    41: FamilyInfo(41, "1-par AS", "1-parametric asymmetric"),
    51: FamilyInfo(51, "1-par AS180", "Rotated 1-parametric asymmetric 180 degree"),
    61: FamilyInfo(61, "1-par AS90", "Rotated 1-parametric asymmetric 90 degree"),
    71: FamilyInfo(71, "1-par AS270", "Rotated 1-parametric asymmetric 270 degree"),
    100: FamilyInfo(100, "NP", "Nonparametric"),
    104: FamilyInfo(104, "Tawn", "Tawn type 1"),
    114: FamilyInfo(114, "Tawn180", "Rotated Tawn type 1 180 degrees"),
    124: FamilyInfo(124, "Tawn90", "Rotated Tawn type 1 90 degrees"),
    134: FamilyInfo(134, "Tawn270", "Rotated Tawn type 1 270 degrees"),
    204: FamilyInfo(204, "Tawn2", "Tawn type 2"),
    214: FamilyInfo(214, "Tawn2_180", "Rotated Tawn type 2 180 degrees"),
    224: FamilyInfo(224, "Tawn2_90", "Rotated Tawn type 2 90 degrees"),
    234: FamilyInfo(234, "Tawn2_270", "Rotated Tawn type 2 270 degrees"),
}

NAME_TO_FAMILY: dict[str, int] = {}
for info in FAMILY_INFO.values():
    NAME_TO_FAMILY[info.short] = info.code
    NAME_TO_FAMILY[info.long] = info.code
NAME_TO_FAMILY["Tawn type 1"] = 104
NAME_TO_FAMILY["Tawn type 2"] = 204
NAME_TO_FAMILY["Rotated Tawn type 2 90 degrees"] = 224
NAME_TO_FAMILY["Rotated Tawn type 2 270 degrees"] = 234

ALL_FAMILIES = tuple(
    code for code in FAMILY_INFO if code not in {100}
)
AUTO_SELECT_FAMILIES = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    13,
    14,
    16,
    23,
    24,
    26,
    33,
    34,
    36,
)
FAST3_FAMILIES = (0, 1, 5)
GAUSSIAN_FAMILIES = (0, 1)
FAMILY_SETS = {
    "all": AUTO_SELECT_FAMILIES,
    "default": AUTO_SELECT_FAMILIES,
    "auto": AUTO_SELECT_FAMILIES,
    "fast": FAST3_FAMILIES,
    "fast3": FAST3_FAMILIES,
    "recommended": FAST3_FAMILIES,
    "gaussian": GAUSSIAN_FAMILIES,
}
ONE_PARAMETER_FAMILIES = {
    1,
    2,
    3,
    4,
    5,
    6,
    13,
    14,
    16,
    23,
    24,
    26,
    33,
    34,
    36,
    41,
    51,
    61,
    71,
}
TWO_PARAMETER_FAMILIES = set(ALL_FAMILIES) - {0} - ONE_PARAMETER_FAMILIES
POSITIVE_DEP_FAMILIES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 19, 20, 104, 114, 204, 214}
NEGATIVE_DEP_FAMILIES = {1, 2, 5, 23, 24, 26, 27, 28, 29, 30, 33, 34, 36, 37, 38, 39, 40, 124, 134, 224, 234}


def resolve_familyset(familyset=None) -> list[int]:
    """Resolve named or explicit family sets for automatic selection."""

    if familyset is None:
        return list(AUTO_SELECT_FAMILIES)
    if isinstance(familyset, str):
        key = familyset.lower()
        if key not in FAMILY_SETS:
            choices = ", ".join(sorted(FAMILY_SETS))
            raise ValueError(f"Unknown familyset {familyset!r}. Use one of: {choices}")
        return list(FAMILY_SETS[key])
    return [family_number(family) for family in familyset]


def family_number(family: int | str) -> int:
    if isinstance(family, np.generic):
        family = family.item()
    if isinstance(family, str):
        if family not in NAME_TO_FAMILY:
            raise ValueError(f"Family not implemented: {family!r}")
        return NAME_TO_FAMILY[family]
    code = int(family)
    if code not in FAMILY_INFO:
        raise ValueError(f"Family not implemented: {family!r}")
    return code


def family_name(family: Any, short: bool = True):
    if np.ndim(family) > 0 and not isinstance(family, str):
        return np.array([family_name(x, short=short) for x in np.asarray(family).ravel()]).reshape(np.asarray(family).shape)
    if isinstance(family, str):
        return family_number(family)
    info = FAMILY_INFO[family_number(family)]
    return info.short if short else info.long


def n_parameters(family: int) -> int:
    family = family_number(family)
    if family == 0:
        return 0
    return 1 if family in ONE_PARAMETER_FAMILIES else 2


def check_bicop(family: int, par: float = 0.0, par2: float = 0.0) -> bool:
    family = family_number(family)
    par = float(par)
    par2 = float(par2)
    if family == 0:
        return True
    if family in {1, 2}:
        if abs(par) >= 1:
            raise ValueError("Gaussian and t parameters must be in (-1, 1)")
        if family == 2 and par2 <= 2:
            raise ValueError("t-copula degrees of freedom must be larger than 2")
    elif family in {3, 13} and not (0 < par <= 28):
        raise ValueError("Clayton parameter must be in (0, 28]")
    elif family in {4, 14} and not (1 <= par <= 17):
        raise ValueError("Gumbel parameter must be in [1, 17]")
    elif family == 5 and (par == 0 or abs(par) > 35):
        raise ValueError("Frank parameter must be non-zero and in [-35, 35]")
    elif family in {6, 16} and not (1 < par <= 30):
        raise ValueError("Joe parameter must be in (1, 30]")
    elif family in {7, 17} and (not (0 < par <= 7) or not (1 <= par2 <= 7)):
        raise ValueError("BB1 parameters must be par in (0, 7], par2 in [1, 7]")
    elif family in {8, 18} and (not (0 < par <= 6) or not (1 <= par2 <= 8)):
        raise ValueError("BB6 parameters must be par in (0, 6], par2 in [1, 8]")
    elif family in {9, 19} and (not (1 <= par <= 6) or not (0 < par2 <= 75)):
        raise ValueError("BB7 parameters must be par in [1, 6], par2 in (0, 75]")
    elif family in {10, 20} and (not (1 <= par <= 8) or not (1e-4 <= par2 <= 1)):
        raise ValueError("BB8 parameters must be par in [1, 8], par2 in [1e-4, 1]")
    elif family in {23, 33} and not (-28 <= par < 0):
        raise ValueError("Rotated Clayton parameter must be in [-28, 0)")
    elif family in {24, 34} and not (-17 <= par <= -1):
        raise ValueError("Rotated Gumbel parameter must be in [-17, -1]")
    elif family in {26, 36} and not (-30 <= par < -1):
        raise ValueError("Rotated Joe parameter must be in [-30, -1)")
    elif family in {27, 37} and (not (-7 <= par < 0) or not (-7 <= par2 <= -1)):
        raise ValueError("Rotated BB1 parameters must be par in [-7, 0), par2 in [-7, -1]")
    elif family in {28, 38} and (not (-6 <= par < 0) or not (-8 <= par2 <= -1)):
        raise ValueError("Rotated BB6 parameters must be par in [-6, 0), par2 in [-8, -1]")
    elif family in {29, 39} and (not (-6 <= par <= -1) or not (-75 <= par2 < 0)):
        raise ValueError("Rotated BB7 parameters must be par in [-6, -1], par2 in [-75, 0)")
    elif family in {30, 40} and (not (-8 <= par <= -1) or not (-1 <= par2 <= -1e-4)):
        raise ValueError("Rotated BB8 parameters must be par in [-8, -1], par2 in [-1, -1e-4]")
    elif family in {104, 114, 204, 214} and (par < 1 or not (0 <= par2 <= 1)):
        raise ValueError("Tawn parameters must be par >= 1 and par2 in [0, 1]")
    elif family in {124, 134, 224, 234} and (par > -1 or not (0 <= par2 <= 1)):
        raise ValueError("Rotated Tawn parameters must be par <= -1 and par2 in [0, 1]")
    elif family in {41, 51} and par <= 0:
        raise ValueError("Asymmetric copula parameter must be positive")
    elif family in {61, 71} and par >= 0:
        raise ValueError("Rotated asymmetric copula parameter must be negative")
    return True


def _debye1(theta: float) -> float:
    if abs(theta) < 1e-7:
        return 1.0 - theta / 4.0 + theta * theta / 36.0

    def integrand(x):
        x = np.asarray(x, dtype=float)
        return np.where(np.abs(x) < 1e-8, 1.0, x / np.expm1(x))

    return legendre_integrate(integrand, 0.0, theta, n=128) / theta


def _frank_tau(theta: float) -> float:
    if abs(theta) < 1e-8:
        return 0.0
    return 1.0 - 4.0 / theta + 4.0 * _debye1(theta) / theta


def _joe_tau(theta: float) -> float:
    if abs(theta - 2.0) < 1e-10:
        return 1.0 - trigamma(2.0)
    param1 = 2.0 / theta + 1.0
    tem = digamma(2.0) - digamma(param1)
    return 1.0 + tem * 2.0 / (2.0 - theta)


def _bb6_tau(theta: float, delta: float) -> float:
    def integrand(t):
        t = np.asarray(t, dtype=float)
        one = np.maximum(1.0 - t, UMIN)
        return -np.log1p(-one**theta) * (1.0 - t - one ** (-theta) + one ** (-theta) * t) / (delta * theta)

    return 1.0 + 4.0 * legendre_integrate(integrand, UMIN, UMAX, n=160)


def _bb7_tau(theta: float, delta: float) -> float:
    def integrand(t):
        t = np.asarray(t, dtype=float)
        one = np.maximum(1.0 - t, UMIN)
        base = np.maximum(1.0 - one**theta, UMIN)
        return (base ** (-delta) - 1.0) / (
            -theta * delta * one ** (theta - 1.0) * base ** (-delta - 1.0)
        )

    return 1.0 + 4.0 * legendre_integrate(integrand, UMIN, UMAX, n=160)


def _bb8_tau(theta: float, delta: float) -> float:
    def integrand(t):
        t = np.asarray(t, dtype=float)
        base = np.maximum(1.0 - t * delta, UMIN)
        denom = (1.0 - delta) ** theta - 1.0
        ratio = np.maximum((base**theta - 1.0) / denom, UMIN)
        return -np.log(ratio) * (
            1.0 - t * delta - base ** (-theta) + base ** (-theta) * t * delta
        ) / (theta * delta)

    return 1.0 + 4.0 * legendre_integrate(integrand, UMIN, UMAX, n=160)


def tawn_a(t, theta: float, par2: float, par3: float):
    t = np.asarray(t, dtype=float)
    ta = (par3 * t) ** theta + (par2 * (1.0 - t)) ** theta
    return (1.0 - par2) * (1.0 - t) + (1.0 - par3) * t + ta ** (1.0 / theta)


def tawn_a_second(t, theta: float, par2: float, par3: float):
    t = np.asarray(t, dtype=float)
    ta = (par3 * t) ** theta + (par2 * (1.0 - t)) ** theta
    d1ta = theta * (
        par3 * (par3 * t) ** (theta - 1.0)
        - par2 * (par2 * (1.0 - t)) ** (theta - 1.0)
    )
    d2ta = theta * (theta - 1.0) * (
        par3**2 * (par3 * t) ** (theta - 2.0)
        + par2**2 * (par2 * (1.0 - t)) ** (theta - 2.0)
    )
    return (1.0 / theta) * (
        (1.0 / theta - 1.0) * ta ** (1.0 / theta - 2.0) * d1ta**2
        + ta ** (1.0 / theta - 1.0) * d2ta
    )


def _tawn_tau(theta: float, par2: float, par3: float) -> float:
    def integrand(t):
        t = np.asarray(t, dtype=float)
        return t * (1.0 - t) * tawn_a_second(t, theta, par2, par3) / tawn_a(t, theta, par2, par3)

    return legendre_integrate(integrand, UMIN, UMAX, n=160)


def _calc_tau_one(family: int, par: float, par2: float = 0.0) -> float:
    family = family_number(family)
    if family == 0:
        return 0.0
    if family in {1, 2}:
        return 2.0 / math.pi * math.asin(par)
    if family in {3, 13}:
        return par / (par + 2.0)
    if family in {4, 14}:
        return 1.0 - 1.0 / par
    if family == 5:
        return _frank_tau(par)
    if family in {6, 16}:
        return _joe_tau(par)
    if family in {7, 17}:
        return 1.0 - 2.0 / (par2 * (par + 2.0))
    if family in {8, 18}:
        return _bb6_tau(par, par2)
    if family in {9, 19}:
        return _bb7_tau(par, par2)
    if family in {10, 20}:
        return _bb8_tau(par, par2)
    if family in {23, 33}:
        return par / (-par + 2.0)
    if family in {24, 34}:
        return -1.0 - 1.0 / par
    if family in {26, 36}:
        return -_joe_tau(-par)
    if family in {27, 37}:
        return -(1.0 - 2.0 / ((-par2) * (-par + 2.0)))
    if family in {28, 38}:
        return -_bb6_tau(-par, -par2)
    if family in {29, 39}:
        return -_bb7_tau(-par, -par2)
    if family in {30, 40}:
        return -_bb8_tau(-par, -par2)
    if family in {104, 114}:
        return _tawn_tau(par, par2, 1.0)
    if family in {204, 214}:
        return _tawn_tau(par, 1.0, par2)
    if family in {124, 134}:
        return -_tawn_tau(-par, par2, 1.0)
    if family in {224, 234}:
        return -_tawn_tau(-par, 1.0, par2)
    raise NotImplementedError(f"Kendall's tau is not implemented for family {family}")


def par_to_tau(family, par, par2=0.0, check_pars: bool = True):
    fam, p, p2 = np.broadcast_arrays(np.atleast_1d(family), np.atleast_1d(par), np.atleast_1d(par2))
    out = np.empty(fam.shape, dtype=float)
    for idx in np.ndindex(fam.shape):
        f = family_number(int(fam[idx]))
        if check_pars:
            check_bicop(f, float(p[idx]), float(p2[idx]))
        out[idx] = _calc_tau_one(f, float(p[idx]), float(p2[idx]))
    return out.item() if out.size == 1 else out


def _calc_par_from_tau(family: int, tau: float) -> float:
    family = family_number(family)
    if abs(tau) > 0.99999:
        raise ValueError("tau is too close to -1 or 1")
    if family == 0:
        return 0.0
    if family in {1, 2}:
        return math.sin(math.pi * tau / 2.0)
    if family in {3, 13}:
        return 2.0 * tau / (1.0 - tau)
    if family in {4, 14}:
        return 1.0 / (1.0 - tau)
    if family == 5:
        if tau == 0:
            return 0.0
        return bisect_root(lambda x: _frank_tau(x) - tau, -35.0, 35.0, tol=1e-8)
    if family in {6, 16}:
        if tau < 0:
            return 1.000001
        return bisect_root(lambda x: _joe_tau(x) - tau, 1.000001, 30.0, tol=1e-8)
    if family in {23, 33}:
        return 2.0 * tau / (1.0 + tau)
    if family in {24, 34}:
        return -1.0 / (1.0 + tau)
    if family in {26, 36}:
        return -bisect_root(lambda x: _joe_tau(x) + tau, 1.000001, 30.0, tol=1e-8)
    raise ValueError("Kendall's tau inversion is only defined for one-parameter families and t")


def tau_to_par(family, tau, check_taus: bool = True):
    fam, tau_arr = np.broadcast_arrays(np.atleast_1d(family), np.atleast_1d(tau))
    out = np.empty(fam.shape, dtype=float)
    for idx in np.ndindex(fam.shape):
        f = family_number(int(fam[idx]))
        if f in TWO_PARAMETER_FAMILIES and f != 2:
            raise ValueError("For two-parameter copulas except t, Kendall's tau cannot be inverted")
        t = float(tau_arr[idx])
        if check_taus:
            if f in {3, 13} and t <= 0:
                raise ValueError(f"{family_name(f, short=False)} cannot be used for tau <= 0")
            if f in {4, 14, 6, 16} and t < 0:
                raise ValueError(f"{family_name(f, short=False)} cannot be used for tau < 0")
            if f == 5 and t == 0:
                raise ValueError("Frank copula cannot be used for tau = 0")
            if f in {23, 33} and t >= 0:
                raise ValueError(f"{family_name(f, short=False)} cannot be used for tau >= 0")
            if f in {24, 34, 26, 36} and t > 0:
                raise ValueError(f"{family_name(f, short=False)} cannot be used for tau > 0")
        out[idx] = _calc_par_from_tau(f, t)
    return out.item() if out.size == 1 else out


def par_to_taildep(family, par, par2=0.0, check_pars: bool = True):
    fam, p, p2 = np.broadcast_arrays(np.atleast_1d(family), np.atleast_1d(par), np.atleast_1d(par2))
    lower = np.zeros(fam.shape, dtype=float)
    upper = np.zeros(fam.shape, dtype=float)
    for idx in np.ndindex(fam.shape):
        f = family_number(int(fam[idx]))
        theta = float(p[idx])
        delta = float(p2[idx])
        if check_pars:
            check_bicop(f, theta, delta)
        if f == 2:
            scipy = require_scipy()
            val = 2.0 * scipy.stats.t.cdf(
                -math.sqrt(delta + 1.0) * math.sqrt((1.0 - theta) / (1.0 + theta)),
                df=delta + 1.0,
            )
            lower[idx] = upper[idx] = val
        elif f == 3:
            lower[idx] = 2.0 ** (-1.0 / theta)
        elif f in {4, 6}:
            upper[idx] = 2.0 - 2.0 ** (1.0 / theta)
        elif f == 7:
            lower[idx] = 2.0 ** (-1.0 / (theta * delta))
            upper[idx] = 2.0 - 2.0 ** (1.0 / delta)
        elif f == 8:
            upper[idx] = 2.0 - 2.0 ** (1.0 / (theta * delta))
        elif f == 9:
            lower[idx] = 2.0 ** (-1.0 / delta)
            upper[idx] = 2.0 - 2.0 ** (1.0 / theta)
        elif f == 10 and delta == 1:
            upper[idx] = 2.0 - 2.0 ** (1.0 / theta)
        elif f == 13:
            upper[idx] = 2.0 ** (-1.0 / theta)
        elif f in {14, 16}:
            lower[idx] = 2.0 - 2.0 ** (1.0 / theta)
        elif f == 17:
            lower[idx] = 2.0 - 2.0 ** (1.0 / delta)
            upper[idx] = 2.0 ** (-1.0 / (theta * delta))
        elif f == 18:
            lower[idx] = 2.0 - 2.0 ** (1.0 / (theta * delta))
        elif f == 19:
            lower[idx] = 2.0 - 2.0 ** (1.0 / theta)
            upper[idx] = 2.0 ** (-1.0 / delta)
        elif f == 20 and delta == 1:
            lower[idx] = 2.0 - 2.0 ** (1.0 / theta)
        elif f == 104:
            upper[idx] = delta + 1.0 - 2.0 * ((0.5 * delta) ** theta + 0.5**theta) ** (1.0 / theta)
        elif f == 114:
            lower[idx] = delta + 1.0 - 2.0 * ((0.5 * delta) ** theta + 0.5**theta) ** (1.0 / theta)
        elif f == 204:
            upper[idx] = 1.0 + delta - 2.0 * (0.5**theta + (0.5 * delta) ** theta) ** (1.0 / theta)
        elif f == 214:
            lower[idx] = 1.0 + delta - 2.0 * (0.5**theta + (0.5 * delta) ** theta) ** (1.0 / theta)
    result = {"lower": lower, "upper": upper}
    if lower.size == 1:
        return {"lower": float(lower.item()), "upper": float(upper.item())}
    return result


def par_to_beta(family, par, par2=0.0):
    from .bicop import bicop_cdf

    val = bicop_cdf(0.5, 0.5, family, par, par2, check_pars=False)
    return 4.0 * np.asarray(val) - 1.0


BiCopName = family_name
BiCopCheck = check_bicop
BiCopPar2Tau = par_to_tau
BiCopTau2Par = tau_to_par
BiCopPar2TailDep = par_to_taildep
BiCopPar2Beta = par_to_beta
