from __future__ import annotations

import numpy as np

from .bicop import (
    BiCop,
    BiCopAIC,
    BiCopBIC,
    BiCopCDF,
    BiCopCompare,
    BiCopCondSim,
    BiCopEst,
    BiCopHfunc,
    BiCopHfunc1,
    BiCopHfunc2,
    BiCopHinv,
    BiCopHinv1,
    BiCopHinv2,
    BiCopIndTest,
    BiCopLambda,
    BiCopLogLik,
    BiCopPDF,
    BiCopSelect,
    BiCopSim,
    BiCopVuongClarke,
    bicop_aic,
    bicop_bic,
    bicop_cdf,
    bicop_compare,
    bicop_condsim,
    bicop_est,
    bicop_hfunc,
    bicop_hfunc1,
    bicop_hfunc2,
    bicop_hinv,
    bicop_hinv1,
    bicop_hinv2,
    bicop_ind_test,
    bicop_lambda,
    bicop_loglik,
    bicop_pdf,
    bicop_select,
    bicop_sim,
)
from .families import (
    ALL_FAMILIES,
    AUTO_SELECT_FAMILIES,
    BiCopCheck,
    BiCopName,
    BiCopPar2Beta,
    BiCopPar2TailDep,
    BiCopPar2Tau,
    BiCopTau2Par,
    FAMILY_SETS,
    FAST3_FAMILIES,
    GAUSSIAN_FAMILIES,
    check_bicop,
    family_name,
    family_number,
    par_to_beta,
    par_to_taildep,
    par_to_tau,
    resolve_familyset,
    tau_to_par,
)
from .rvine import (
    BetaMatrix,
    C2RVine,
    D2RVine,
    DissmannEdge,
    DissmannVine,
    RVineAIC,
    RVineBIC,
    RVineCopSelect,
    RVineLogLik,
    RVineMatrix,
    RVineMatrixNormalize,
    RVineMatrixSample,
    RVinePDF,
    RVinePar2Beta,
    RVinePar2Tau,
    RVineSim,
    RVineStructureSelect,
    TauMatrix as RVineTauMatrix,
    create_max_mat,
    dissmann_loglik,
    dissmann_structure_select,
    needed_cond_distr,
    rvine_aic,
    rvine_bic,
    rvine_loglik,
    rvine_pdf,
    rvine_sim,
)
from .stats import EmpCDF, FredMDData, TauMatrix, as_copuladata, load_fred_md_csv, pairs_copuladata, pobs

__version__ = "0.1.0"


def _not_ported(name: str):
    def inner(*_, **__):
        raise NotImplementedError(
            f"{name} is not yet ported to native Python. Core CDF/PDF/h-function, "
            "simulation, estimation, and R-vine likelihood APIs are available."
        )

    inner.__name__ = name
    return inner


BiCopDeriv = _not_ported("BiCopDeriv")
BiCopDeriv2 = _not_ported("BiCopDeriv2")
BiCopHfuncDeriv = _not_ported("BiCopHfuncDeriv")
BiCopHfuncDeriv2 = _not_ported("BiCopHfuncDeriv2")
BiCopGofTest = _not_ported("BiCopGofTest")
BiCopKDE = _not_ported("BiCopKDE")
BiCopChiPlot = _not_ported("BiCopChiPlot")
BiCopKPlot = _not_ported("BiCopKPlot")
BiCopMetaContour = _not_ported("BiCopMetaContour")
RVineCDF = _not_ported("RVineCDF")
RVineMLE = _not_ported("RVineMLE")
RVineSeqEst = _not_ported("RVineSeqEst")
RVineGrad = _not_ported("RVineGrad")
RVineHessian = _not_ported("RVineHessian")
RVineStdError = _not_ported("RVineStdError")
RVinePIT = _not_ported("RVinePIT")
RVineGofTest = _not_ported("RVineGofTest")
RVineVuongTest = _not_ported("RVineVuongTest")
RVineClarkeTest = _not_ported("RVineClarkeTest")
RVineTreePlot = _not_ported("RVineTreePlot")
plot_BiCop = _not_ported("plot.BiCop")
plot_RVineMatrix = _not_ported("plot.RVineMatrix")
contour_RVineMatrix = _not_ported("contour.RVineMatrix")


def RVineCor2pcor(cor):
    cor = np.asarray(cor, dtype=float)
    inv = np.linalg.inv(cor)
    d = np.sqrt(np.diag(inv))
    return -inv / np.outer(d, d)


def RVinePcor2cor(pcor):
    pcor = np.asarray(pcor, dtype=float)
    inv = -pcor.copy()
    np.fill_diagonal(inv, 1.0)
    cor = np.linalg.inv(inv)
    d = np.sqrt(np.diag(cor))
    return cor / np.outer(d, d)


def copulaFromFamilyIndex(family, par=None, par2=0.0):
    return BiCop(family, par, par2)


def vineCopula(*args, **kwargs):
    return RVineMatrix(*args, **kwargs)


def _factory(family):
    return lambda par=None, par2=0.0, **_: BiCop(family, par, par2)


BB1Copula = _factory(7)
surBB1Copula = _factory(17)
r90BB1Copula = _factory(27)
r270BB1Copula = _factory(37)
BB6Copula = _factory(8)
surBB6Copula = _factory(18)
r90BB6Copula = _factory(28)
r270BB6Copula = _factory(38)
BB7Copula = _factory(9)
surBB7Copula = _factory(19)
r90BB7Copula = _factory(29)
r270BB7Copula = _factory(39)
BB8Copula = _factory(10)
surBB8Copula = _factory(20)
r90BB8Copula = _factory(30)
r270BB8Copula = _factory(40)
joeBiCopula = _factory(6)
surJoeBiCopula = _factory(16)
r90JoeBiCopula = _factory(26)
r270JoeBiCopula = _factory(36)
surClaytonCopula = _factory(13)
r90ClaytonCopula = _factory(23)
r270ClaytonCopula = _factory(33)
surGumbelCopula = _factory(14)
r90GumbelCopula = _factory(24)
r270GumbelCopula = _factory(34)
tawnT1Copula = _factory(104)
surTawnT1Copula = _factory(114)
r90TawnT1Copula = _factory(124)
r270TawnT1Copula = _factory(134)
tawnT2Copula = _factory(204)
surTawnT2Copula = _factory(214)
r90TawnT2Copula = _factory(224)
r270TawnT2Copula = _factory(234)


__all__ = [name for name in globals() if not name.startswith("_")]
