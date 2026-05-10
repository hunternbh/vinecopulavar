# vinecopulavar

Native Python port of core functionality from
[`tnagler/VineCopula`](https://github.com/tnagler/VineCopula). I added extra functionality for a a dissmann-style searched using kendall's tau best distributions for the bivariate copulas. Please see the examples below.

If this code has benefitted you, please cite my paper. Thank you so much!:
Ng, H. (2026). Vine Copula VAR. Zicklin School of Business, Baruch College, City University of New York.

The implementation lives in `src/vinecopula`. The original R package clone is
kept in `source` only as reference material. This package keeps R-style aliases
such as `BiCopPDF`, `BiCopHfunc1`, `RVineStructureSelect`, and `pobs`, while
also exposing Python-style helpers.

This is not an R bridge and does not compile the original C sources. NumPy is
used throughout. SciPy is optional and is only needed for Student-t distribution
functions and numerical MLE paths.

See `PORTING_STATUS.md` for the exact coverage matrix and known limitations.

## Install

From this workspace:

```powershell
pip install -e .
```

Or use directly from the checkout:

```python
import sys
sys.path.insert(0, "src")
```

## Main Functionality

- Bivariate copula objects: `BiCop`
- Bivariate CDF/PDF/log-likelihood: `BiCopCDF`, `BiCopPDF`, `BiCopLogLik`
- h-functions and inverse h-functions: `BiCopHfunc`, `BiCopHfunc1`,
  `BiCopHfunc2`, `BiCopHinv`, `BiCopHinv1`, `BiCopHinv2`
- Bivariate simulation and conditional simulation: `BiCopSim`,
  `BiCopCondSim`
- Kendall's tau and tail dependence: `BiCopTau2Par`, `BiCopPar2Tau`,
  `BiCopPar2TailDep`, `BiCopPar2Beta`
- Bivariate family selection: `BiCopSelect`, including named candidate sets
  such as `familyset="all"`, `"recommended"`/`"fast3"`, and `"gaussian"`
- Pseudo-observations and empirical CDF helpers: `pobs`, `EmpCDF`,
  `TauMatrix`
- Fixed-structure R-vine objects and likelihood/simulation: `RVineMatrix`,
  `RVineLogLik`, `RVinePDF`, `RVineSim`, `RVineAIC`, `RVineBIC`
- Dissmann-style sequential R-vine structure search:
  `RVineStructureSelect` / `dissmann_structure_select`
- FRED-MD CSV loading with support for the `Transform:` row:
  `load_fred_md_csv`

## Available Bivariate Copula Families

These are the family codes understood by the port. Codes match VineCopula.

| Code | Short | Family | Parameters | Automatic `BiCopSelect` |
|---:|---|---|---|---|
| `0` | `I` | Independence | none | yes |
| `1` | `N` | Gaussian | `par` correlation in `(-1, 1)` | yes |
| `2` | `t` | Student-t | `par` correlation, `par2` df `> 2` | partial, needs fixed/default df |
| `3` | `C` | Clayton | `par > 0` | yes |
| `4` | `G` | Gumbel | `par >= 1` | yes |
| `5` | `F` | Frank | `par != 0` | yes |
| `6` | `J` | Joe | `par > 1` | yes |
| `7` | `BB1` | BB1 | `par`, `par2` | manual parameters |
| `8` | `BB6` | BB6 | `par`, `par2` | manual parameters |
| `9` | `BB7` | BB7 | `par`, `par2` | manual parameters |
| `10` | `BB8` | BB8 / Frank-Joe | `par`, `par2` | manual parameters |
| `13` | `SC` | Survival Clayton | `par > 0` | yes |
| `14` | `SG` | Survival Gumbel | `par >= 1` | yes |
| `16` | `SJ` | Survival Joe | `par > 1` | yes |
| `17` | `SBB1` | Survival BB1 | `par`, `par2` | manual parameters |
| `18` | `SBB6` | Survival BB6 | `par`, `par2` | manual parameters |
| `19` | `SBB7` | Survival BB7 | `par`, `par2` | manual parameters |
| `20` | `SBB8` | Survival BB8 | `par`, `par2` | manual parameters |
| `23` | `C90` | Clayton rotated 90 degrees | negative `par` | yes |
| `24` | `G90` | Gumbel rotated 90 degrees | negative `par` | yes |
| `26` | `J90` | Joe rotated 90 degrees | negative `par` | yes |
| `27` | `BB1_90` | BB1 rotated 90 degrees | negative `par`, `par2` | manual parameters |
| `28` | `BB6_90` | BB6 rotated 90 degrees | negative `par`, `par2` | manual parameters |
| `29` | `BB7_90` | BB7 rotated 90 degrees | negative `par`, `par2` | manual parameters |
| `30` | `BB8_90` | BB8 rotated 90 degrees | negative `par`, `par2` | manual parameters |
| `33` | `C270` | Clayton rotated 270 degrees | negative `par` | yes |
| `34` | `G270` | Gumbel rotated 270 degrees | negative `par` | yes |
| `36` | `J270` | Joe rotated 270 degrees | negative `par` | yes |
| `37` | `BB1_270` | BB1 rotated 270 degrees | negative `par`, `par2` | manual parameters |
| `38` | `BB6_270` | BB6 rotated 270 degrees | negative `par`, `par2` | manual parameters |
| `39` | `BB7_270` | BB7 rotated 270 degrees | negative `par`, `par2` | manual parameters |
| `40` | `BB8_270` | BB8 rotated 270 degrees | negative `par`, `par2` | manual parameters |
| `104` | `Tawn` | Tawn type 1 | `par`, `par2` | manual parameters |
| `114` | `Tawn180` | Tawn type 1 rotated 180 degrees | `par`, `par2` | manual parameters |
| `124` | `Tawn90` | Tawn type 1 rotated 90 degrees | negative `par`, `par2` | manual parameters |
| `134` | `Tawn270` | Tawn type 1 rotated 270 degrees | negative `par`, `par2` | manual parameters |
| `204` | `Tawn2` | Tawn type 2 | `par`, `par2` | manual parameters |
| `214` | `Tawn2_180` | Tawn type 2 rotated 180 degrees | `par`, `par2` | manual parameters |
| `224` | `Tawn2_90` | Tawn type 2 rotated 90 degrees | negative `par`, `par2` | manual parameters |
| `234` | `Tawn2_270` | Tawn type 2 rotated 270 degrees | negative `par`, `par2` | manual parameters |

The one-parameter families, plus Student-t with a default/fixed degrees of
freedom, are available for Kendall-tau inversion and therefore for automatic
`BiCopSelect`. Two-parameter families such as BB1, BB6, BB7, BB8, and Tawn are
available for evaluation, h-functions, simulation, and fixed-parameter vines,
but automatic estimation from Kendall's tau is not unique, so those are not
selected automatically unless a future MLE selector is added.

The code also has family-name entries for the asymmetric families `41`, `51`,
`61`, and `71`, but their full CDF/PDF/h-function implementation is not yet
complete in the native Python port. They should not be used in model selection.

## Common Family Sets

Default automatic bivariate selection uses every family that can currently be
estimated from Kendall's tau:

```python
familyset = "all"
# Equivalent to:
# [0, 1, 2, 3, 4, 5, 6, 13, 14, 16, 23, 24, 26, 33, 34, 36]
```

For fast high-dimensional screening, the recommended three-family set is:

```python
familyset = "recommended"   # alias: "fast3"
# Equivalent to: [0, 1, 5]
# Independence, Gaussian, Frank
```

For Gaussian-only screening:

```python
familyset = "gaussian"
# Equivalent to: [0, 1]
```

You can always pass an explicit custom list:

```python
familyset = [0, 1, 5, 3, 13]
```

For manually specified pair-copulas you can use two-parameter families:

```python
from vinecopula import BiCop, BiCopPDF

bb1 = BiCop(7, par=1.2, par2=2.0)
density = BiCopPDF([0.2, 0.5], [0.4, 0.7], bb1)
```

## Bivariate Examples

Create a copula and evaluate it:

```python
from vinecopula import BiCop, BiCopCDF, BiCopPDF, BiCopHfunc1, BiCopHinv1

cop = BiCop(family=3, par=2.0)  # Clayton

cdf = BiCopCDF(0.4, 0.6, cop)
pdf = BiCopPDF(0.4, 0.6, cop)

z = BiCopHinv1(0.25, 0.75, cop)
check = BiCopHfunc1(0.25, z, cop)
```

Simulate and select a family:

```python
from vinecopula import BiCopSim, BiCopSelect

u = BiCopSim(500, cop, random_state=7)
fit = BiCopSelect(u[:, 0], u[:, 1], familyset="all", selectioncrit="AIC")

print(fit.family, fit.familyname, fit.par, fit.tau)
```

Use the fast preset when you are screening many bivariate pairs:

```python
fast_fit = BiCopSelect(u[:, 0], u[:, 1], familyset="recommended")
```

Convert Kendall's tau to a parameter:

```python
from vinecopula import BiCopTau2Par, BiCopPar2Tau

rho = BiCopTau2Par(1, 0.5)      # Gaussian rho
tau = BiCopPar2Tau(1, rho)
```

## Pseudo-Observations

Copula models expect uniform margins. Use `pobs` to rank-transform raw data:

```python
import numpy as np
from vinecopula import pobs

x = np.random.default_rng(1).normal(size=(200, 5))
u = pobs(x)
```

## Fixed-Structure R-Vine Example

```python
import numpy as np
from vinecopula import RVineMatrix, RVineLogLik, RVineSim

matrix = np.array([[2, 0], [1, 1]])
family = np.array([[0, 0], [3, 0]])
par = np.array([[0.0, 0.0], [2.0, 0.0]])

rvm = RVineMatrix(matrix, family, par)
u = RVineSim(250, rvm, random_state=42)
loglik = RVineLogLik(u, rvm)["loglik"]
```

## Dissmann-Style Vine Structure Search

`RVineStructureSelect` builds each tree using the Dissmann maximum spanning
tree algorithm. Edge weights are `abs(Kendall tau)` of the relevant conditional
pseudo-observations. After an edge enters the maximum spanning tree, its
bivariate distribution is selected by `BiCopSelect`: for each candidate family
in `familyset`, the parameter is estimated by Kendall-tau inversion, the
log-likelihood is evaluated, and the family with the best AIC or BIC is kept.

```python
import numpy as np
from vinecopula import RVineStructureSelect, RVineLogLik, pobs

rng = np.random.default_rng(42)
x = rng.normal(size=(400, 10))
u = pobs(x)

model = RVineStructureSelect(
    u,
    as_pobs_data=False,
    familyset="all",
    selectioncrit="AIC",
)

print(model)
print(model.edge_table()[:5])
print(RVineLogLik(u, model)["loglik"])
```

For high-dimensional data, use `n_trees` to fit only the first few strongest
vine tree layers. Full untruncated selection is supported, but a 100+ variable
panel contains thousands of pair-copulas. `trunc_lvl` is still accepted as a
backward-compatible alias for `n_trees`.

```python
model = RVineStructureSelect(
    u,
    as_pobs_data=False,
    familyset="recommended",
    n_trees=10,
    tau_threshold=0.02,
)
```

The returned object is a native `DissmannVine`. It stores selected edges as a
graph rather than forcing every selected structure into the legacy R matrix
layout:

```python
rows = model.edge_table()
first_edge = rows[0]
print(model.n_layers, model.n_trees, model.n_edges)
pointwise_loglik = model.loglik(u, separate=True)["loglik"]
```

## FRED-MD Example

The FRED-MD CSV has a `sasdate` column and a `Transform:` row. The loader reads
that row, applies the standard McCracken-Ng transformations, drops very sparse
columns, optionally fills missing values, and returns pseudo-observations ready
for copula modeling.

```python
from vinecopula import load_fred_md_csv, RVineStructureSelect

fred = load_fred_md_csv(
    r"C:\Users\hia_n\Desktop\vine-copular-var\fred-data\2026-03-MD.csv",
    apply_transforms=True,
    as_pobs_data=True,
    start_date="1990-01-01",
    max_missing=0.25,
    fill="ffill_bfill",
)

model = RVineStructureSelect(
    fred.data,
    names=fred.names,
    as_pobs_data=False,
    familyset="recommended",
    n_trees=10,
    tau_threshold=0.02,
)

print(fred.data.shape)
print(model, model.n_layers, model.n_edges)
print(model.edge_table()[:10])
```

For an untruncated fit, omit `n_trees`. That is the closest analogue to the
full Dissmann search, but it can be slow on FRED-MD scale data.

## Practical Notes

- Use `pobs` or `load_fred_md_csv(..., as_pobs_data=True)` before fitting
  copulas to raw macro or financial data.
- Use `familyset="recommended"` or `"fast3"` for quick 128-variable FRED-MD
  experiments. Use `familyset="all"` when you want the broader automatic
  candidate set and can afford the extra runtime.
- Use `selectioncrit="BIC"` for more parsimonious automatic bivariate selection.
- Use `n_trees` for high-dimensional vines. A 128-variable full vine has
  `128 * 127 / 2 = 8128` pair-copula edges.
- Student-t copulas require SciPy for distribution calculations.
- BB and Tawn families currently use numerical derivatives/inversion in several
  paths, so they are slower than Gaussian, Clayton, Gumbel, Frank, and Joe.
