# VineCopula Porting Status

This is a Python-native port, not a bridge to the R package.

## Implemented

- Bivariate copula objects and validation: `BiCop`, `BiCopCheck`, `BiCopName`
- Family-name mappings for all VineCopula family codes
- CDF/PDF/log-likelihood for independence, Gaussian, Clayton, Gumbel, Frank,
  Joe, BB1, BB6, BB7, BB8, Tawn type 1/2, and their standard rotations
- Numerical h-functions and inverse h-functions: `BiCopHfunc*`, `BiCopHinv*`
- Bivariate simulation: `BiCopSim`, `BiCopCondSim`
- Kendall's tau conversions and tail dependence: `BiCopPar2Tau`,
  `BiCopTau2Par`, `BiCopPar2TailDep`
- Pseudo-observations and empirical CDF: `pobs`, `EmpCDF`
- R-vine matrix container plus core traversal: `RVineMatrix`, `RVineLogLik`,
  `RVinePDF`, `RVineSim`, `RVineAIC`, `RVineBIC`
- Dissmann-style sequential R-vine structure search:
  `RVineStructureSelect`, `dissmann_structure_select`
- Basic bivariate estimation and family selection: `BiCopEst`, `BiCopSelect`
- FRED-MD CSV loading with transformation-row support: `load_fred_md_csv`

## Partially Implemented

- Student-t copulas require SciPy at runtime for CDF/PDF/h-functions.
- MLE uses SciPy when installed; otherwise one-parameter estimation falls back
  to inversion of Kendall's tau.
- Goodness-of-fit and comparison helpers return native approximations where
  practical, but are not byte-for-byte ports of the R package's tests.
- `RVineStructureSelect` returns a native `DissmannVine` graph model instead of
  reconstructing every selected structure as the legacy R matrix layout.

## Not Yet Ported

- Plotting functions that depend on R graphics/lattice are exposed as
  placeholders with clear `NotImplementedError` messages.
- The deprecated R `copula` package compatibility constructors are represented
  by lightweight `BiCop` factories rather than S4 copula classes.
- Analytic derivatives from the C files are not fully ported; numerical
  derivatives are used for the high-level Python API.
