from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ._utils import empirical_kendall_tau, ensure_square_matrix
from .bicop import (
    BiCop,
    bicop_hfunc1,
    bicop_hfunc2,
    bicop_hinv1,
    bicop_loglik,
    bicop_pdf,
    bicop_select,
)
from .families import (
    TWO_PARAMETER_FAMILIES,
    n_parameters,
    par_to_beta,
    par_to_taildep,
    par_to_tau,
    resolve_familyset,
)
from .stats import pobs


def reorder_rvine_matrix(matrix, old_order=None):
    mat = np.array(matrix, dtype=int, copy=True)
    n = mat.shape[0]
    if old_order is None:
        old_order = np.diag(mat).copy()
    mapping = {int(old_order[i]): n - i for i in range(n)}
    out = mat.copy()
    for old, new in mapping.items():
        out[mat == old] = new
    return out


def create_max_mat(matrix):
    original = np.asarray(matrix, dtype=int)
    maxmat = reorder_rvine_matrix(original)
    n = maxmat.shape[0]
    for j in range(n - 1):
        for i in range(n - 2, j - 1, -1):
            maxmat[i, j] = int(np.max(maxmat[i : i + 2, j]))
    tmp = maxmat.copy()
    old_sort = np.diag(original)[::-1]
    for i in range(1, n + 1):
        maxmat[tmp == i] = old_sort[i - 1]
    return maxmat


def needed_cond_distr(matrix):
    vine = reorder_rvine_matrix(matrix)
    maxmat = create_max_mat(vine)
    d = vine.shape[0]
    direct = np.zeros((d, d), dtype=bool)
    indirect = np.zeros((d, d), dtype=bool)
    direct[1:d, 0] = True
    for i0 in range(1, d - 1):
        v = d - i0
        bw = maxmat[i0:d, 0:i0] == v
        is_direct = vine[i0:d, 0:i0] == v
        indirect[i0:d, i0] = np.any(bw & (~is_direct), axis=1)
        direct[i0:d, i0] = True
        direct[i0, i0] = bool(np.any(bw[0, :] & is_direct[0, :]))
    return {"direct": direct, "indirect": indirect}


def is_dvine(matrix):
    mat = reorder_rvine_matrix(matrix.Matrix if isinstance(matrix, RVineMatrix) else matrix)
    d = mat.shape[0]
    nums = list(np.diag(mat)[: d - 1]) + list(mat[d - 1, : d - 1])
    counts = sorted(np.unique(nums, return_counts=True)[1])
    return counts.count(1) == 2 and max(counts) == 2


def is_cvine(matrix):
    mat = reorder_rvine_matrix(matrix.Matrix if isinstance(matrix, RVineMatrix) else matrix)
    d = mat.shape[0]
    if d < 4:
        return True
    return all(len(np.unique(mat[d - tree + 1, : d - tree])) == 1 for tree in range(2, d - 1))


@dataclass
class RVineMatrix:
    Matrix: np.ndarray
    family: np.ndarray | None = None
    par: np.ndarray | None = None
    par2: np.ndarray | None = None
    names: list[str] | None = None
    check_pars: bool = True
    MaxMat: np.ndarray = field(init=False)
    CondDistr: dict[str, np.ndarray] = field(init=False)
    type: str = field(init=False)
    tau: np.ndarray = field(init=False)
    beta: np.ndarray = field(init=False)
    taildep: dict[str, np.ndarray] = field(init=False)

    def __post_init__(self):
        self.Matrix = ensure_square_matrix(self.Matrix, "Matrix").astype(int)
        d = self.Matrix.shape[0]
        if self.family is None:
            self.family = np.zeros((d, d), dtype=int)
        else:
            self.family = ensure_square_matrix(self.family, "family").astype(int)
        if self.par is None:
            self.par = np.zeros((d, d), dtype=float)
        else:
            self.par = ensure_square_matrix(self.par, "par").astype(float)
        if self.par2 is None:
            self.par2 = np.zeros((d, d), dtype=float)
        else:
            self.par2 = ensure_square_matrix(self.par2, "par2").astype(float)
        if self.names is not None and len(self.names) != d:
            raise ValueError("names must have one entry per dimension")
        self.MaxMat = create_max_mat(self.Matrix)
        self.CondDistr = needed_cond_distr(self.Matrix)
        self.type = "C-vine" if is_cvine(self) else "D-vine" if is_dvine(self) else "R-vine"
        self.tau = np.zeros((d, d), dtype=float)
        self.beta = np.zeros((d, d), dtype=float)
        lower = np.zeros((d, d), dtype=float)
        upper = np.zeros((d, d), dtype=float)
        for i in range(d):
            for j in range(i):
                fam = int(self.family[i, j])
                if fam == 0:
                    continue
                self.tau[i, j] = par_to_tau(fam, self.par[i, j], self.par2[i, j], check_pars=False)
                try:
                    self.beta[i, j] = par_to_beta(fam, self.par[i, j], self.par2[i, j])
                except Exception:
                    self.beta[i, j] = np.nan
                td = par_to_taildep(fam, self.par[i, j], self.par2[i, j], check_pars=False)
                lower[i, j] = td["lower"]
                upper[i, j] = td["upper"]
        self.taildep = {"lower": lower, "upper": upper}

    def __len__(self):
        return self.Matrix.shape[0]

    def normalize(self):
        old_order = np.diag(self.Matrix)
        matrix = reorder_rvine_matrix(self.Matrix)
        names = None if self.names is None else list(np.asarray(self.names, dtype=object)[old_order[::-1] - 1])
        return RVineMatrix(matrix, self.family, self.par, self.par2, names=names, check_pars=False)

    def __repr__(self):
        return f"RVineMatrix(type={self.type!r}, dim={len(self)})"


def RVineMatrixNormalize(rvm: RVineMatrix) -> RVineMatrix:
    return rvm.normalize()


def _prepare_rvm_and_data(data, rvm: RVineMatrix):
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(rvm):
        raise ValueError("data dimension does not match RVM")
    order = np.diag(rvm.Matrix)
    if not np.array_equal(order, np.arange(len(rvm), 0, -1)):
        rvm = rvm.normalize()
        data = data[:, order[::-1] - 1]
    return data, rvm


def rvine_loglik(
    data,
    RVM: RVineMatrix,
    par=None,
    par2=None,
    *,
    separate: bool = False,
    calculate_V: bool = True,
    check_pars: bool = True,
):
    if isinstance(RVM, DissmannVine):
        return dissmann_loglik(data, RVM, separate=separate)
    if par is not None:
        RVM = RVineMatrix(RVM.Matrix, RVM.family, par, RVM.par2, RVM.names, check_pars=False)
    if par2 is not None:
        RVM = RVineMatrix(RVM.Matrix, RVM.family, RVM.par, par2, RVM.names, check_pars=False)
    data, rvm = _prepare_rvm_and_data(data, RVM)
    n_obs, d = data.shape
    vdirect = np.zeros((d, d, n_obs), dtype=float)
    vindirect = np.zeros((d, d, n_obs), dtype=float)
    value = np.zeros((d, d, n_obs), dtype=float)
    for i in range(d):
        vdirect[d - 1, i, :] = data[:, d - 1 - i]

    for i in range(d - 2, -1, -1):
        for k in range(d - 1, i, -1):
            fam = int(rvm.family[k, i])
            theta = float(rvm.par[k, i])
            nu = float(rvm.par2[k, i])
            m = int(rvm.MaxMat[k, i])
            m_idx = d - m
            if int(rvm.Matrix[k, i]) == m:
                x = vdirect[k, m_idx, :]
                y = vdirect[k, i, :]
                value[k, i, :] = bicop_loglik(x, y, fam, theta, nu, separate=True, check_pars=False)
                if rvm.CondDistr["direct"][k - 1, i]:
                    vdirect[k - 1, i, :] = bicop_hfunc1(x, y, fam, theta, nu, check_pars=False)
                if rvm.CondDistr["indirect"][k - 1, i]:
                    vindirect[k - 1, i, :] = bicop_hfunc2(x, y, fam, theta, nu, check_pars=False)
            else:
                x = vindirect[k, m_idx, :]
                y = vdirect[k, i, :]
                value[k, i, :] = bicop_loglik(x, y, fam, theta, nu, separate=True, check_pars=False)
                if rvm.CondDistr["direct"][k - 1, i]:
                    vdirect[k - 1, i, :] = bicop_hfunc1(x, y, fam, theta, nu, check_pars=False)
                if rvm.CondDistr["indirect"][k - 1, i]:
                    vindirect[k - 1, i, :] = bicop_hfunc2(x, y, fam, theta, nu, check_pars=False)

    pointwise = np.sum(value, axis=(0, 1))
    out = {"loglik": pointwise if separate else float(np.sum(pointwise))}
    if calculate_V:
        out["V"] = {"direct": vdirect, "indirect": vindirect, "value": value if separate else np.sum(value, axis=2)}
    return out


def rvine_pdf(data, RVM: RVineMatrix, par=None, par2=None, *, check_pars: bool = True):
    if isinstance(RVM, DissmannVine):
        return np.exp(dissmann_loglik(data, RVM, separate=True)["loglik"])
    return np.exp(rvine_loglik(data, RVM, par, par2, separate=True, calculate_V=False, check_pars=check_pars)["loglik"])


def rvine_aic(data, RVM: RVineMatrix, par=None, par2=None):
    if isinstance(RVM, DissmannVine):
        like = dissmann_loglik(data, RVM)
        npar = sum(n_parameters(edge.family) for tree in RVM.edges for edge in tree)
        return {"AIC": float(-2.0 * like["loglik"] + 2.0 * npar)}
    like = rvine_loglik(data, RVM, par, par2)
    fam = RVM.family
    npar_pair = np.vectorize(n_parameters)(fam)
    npar_pair[fam == 0] = 0
    pair = -2.0 * like["V"]["value"] + 2.0 * npar_pair
    return {"AIC": float(-2.0 * like["loglik"] + 2.0 * np.sum(npar_pair)), "pair.AIC": pair}


def rvine_bic(data, RVM: RVineMatrix, par=None, par2=None):
    data_arr = np.asarray(data)
    n = data_arr.shape[0] if data_arr.ndim > 1 else 1
    if isinstance(RVM, DissmannVine):
        like = dissmann_loglik(data, RVM)
        npar = sum(n_parameters(edge.family) for tree in RVM.edges for edge in tree)
        return {"BIC": float(-2.0 * like["loglik"] + math.log(max(n, 1)) * npar)}
    like = rvine_loglik(data, RVM, par, par2)
    fam = RVM.family
    npar_pair = np.vectorize(n_parameters)(fam)
    npar_pair[fam == 0] = 0
    pair = -2.0 * like["V"]["value"] + math.log(max(n, 1)) * npar_pair
    return {"BIC": float(-2.0 * like["loglik"] + math.log(max(n, 1)) * np.sum(npar_pair)), "pair.BIC": pair}


def rvine_sim(N: int, RVM: RVineMatrix, U=None, *, random_state=None):
    if not isinstance(RVM, RVineMatrix):
        raise TypeError("RVM must be an RVineMatrix")
    d = len(RVM)
    order = np.diag(RVM.Matrix)
    rvm = RVM.normalize()
    rng = np.random.default_rng(random_state)
    if U is None:
        U2 = rng.uniform(size=(int(N), d))
    else:
        U = np.asarray(U, dtype=float)
        if U.ndim == 1:
            U = U.reshape(1, -1)
        U2 = U[:, order[::-1] - 1]
        N = U2.shape[0]

    family = rvm.family[::-1, ::-1]
    par = rvm.par[::-1, ::-1]
    par2 = rvm.par2[::-1, ::-1]
    maxmat = rvm.MaxMat[::-1, ::-1]
    mat = rvm.Matrix[::-1, ::-1]
    cindirect = rvm.CondDistr["indirect"][::-1, ::-1]
    out = np.zeros((int(N), d), dtype=float)

    for row in range(int(N)):
        vdirect = np.zeros((d, d), dtype=float)
        vindirect = np.zeros((d, d), dtype=float)
        for i in range(d):
            vdirect[i, i] = U2[row, i]
        vindirect[0, 0] = vdirect[0, 0]
        for i in range(1, d):
            for k in range(i - 1, -1, -1):
                fam = int(family[k, i])
                theta = float(par[k, i])
                nu = float(par2[k, i])
                m = int(maxmat[k, i])
                if int(mat[k, i]) == m:
                    cond = vdirect[k, m - 1]
                else:
                    cond = vindirect[k, m - 1]
                vdirect[k, i] = bicop_hinv1(cond, vdirect[k + 1, i], fam, theta, nu, check_pars=False)
                if i + 1 < d and cindirect[k + 1, i]:
                    if int(mat[k, i]) == m:
                        vindirect[k + 1, i] = bicop_hfunc2(vdirect[k, m - 1], vdirect[k, i], fam, theta, nu, check_pars=False)
                    else:
                        vindirect[k + 1, i] = bicop_hfunc2(vindirect[k, m - 1], vdirect[k, i], fam, theta, nu, check_pars=False)
        out[row, :] = vdirect[0, :]

    ix = np.argsort(order[::-1])
    return out[:, ix]


def rvine_par_to_tau(RVM: RVineMatrix):
    return RVM.tau.copy()


def rvine_par_to_beta(RVM: RVineMatrix):
    return RVM.beta.copy()


def BetaMatrix(RVM: RVineMatrix):
    return rvine_par_to_beta(RVM)


def TauMatrix(RVM: RVineMatrix):
    return rvine_par_to_tau(RVM)


def C2RVine(order, family=None, par=None, par2=None):
    order = np.asarray(order, dtype=int)
    d = len(order)
    mat = np.zeros((d, d), dtype=int)
    np.fill_diagonal(mat, order[::-1])
    for j in range(d - 1):
        mat[j + 1 :, j] = order[j]
    return RVineMatrix(mat, family, par, par2, check_pars=False)


def D2RVine(order, family=None, par=None, par2=None):
    order = np.asarray(order, dtype=int)
    d = len(order)
    mat = np.zeros((d, d), dtype=int)
    np.fill_diagonal(mat, order[::-1])
    for j in range(d - 1):
        mat[d - 1, j] = order[j]
        for i in range(j + 1, d - 1):
            mat[i, j] = order[i]
    return RVineMatrix(mat, family, par, par2, check_pars=False)


def RVineMatrixSample(d: int, natural_order: bool = True, random_state=None):
    rng = np.random.default_rng(random_state)
    order = np.arange(1, d + 1)
    if not natural_order:
        rng.shuffle(order)
    return D2RVine(order)


@dataclass
class _DissmannNode:
    node_id: int
    complete: frozenset[int]
    cond_data: dict[tuple[int, frozenset[int]], np.ndarray]
    parent_ids: frozenset[int] = field(default_factory=frozenset)


@dataclass
class DissmannEdge:
    tree: int
    edge_id: int
    left_node: int
    right_node: int
    new_node: int
    conditioned: tuple[int, int]
    conditioning: tuple[int, ...]
    tau: float
    weight: float
    family: int
    par: float
    par2: float
    loglik: float
    criterion: float


@dataclass
class DissmannVine:
    dim: int
    edges: list[list[DissmannEdge]]
    names: list[str] | None = None
    familyset: list[int] | None = None
    selectioncrit: str = "AIC"
    trunc_lvl: int | None = None
    n_trees: int | None = None
    tau_threshold: float = 0.0
    nobs: int | None = None

    @property
    def type(self) -> str:
        return "R-vine"

    @property
    def n_edges(self) -> int:
        return sum(len(tree) for tree in self.edges)

    @property
    def n_layers(self) -> int:
        return len(self.edges)

    def edge_table(self):
        rows = []
        for tree in self.edges:
            for edge in tree:
                left, right = edge.conditioned
                cond = ",".join(self._name(i) for i in edge.conditioning)
                rows.append(
                    {
                        "tree": edge.tree,
                        "edge": edge.edge_id,
                        "left": self._name(left),
                        "right": self._name(right),
                        "conditioning": cond,
                        "family": edge.family,
                        "par": edge.par,
                        "par2": edge.par2,
                        "tau": edge.tau,
                        "weight": edge.weight,
                        "loglik": edge.loglik,
                        self.selectioncrit: edge.criterion,
                    }
                )
        return rows

    def _name(self, idx: int) -> str:
        return self.names[idx] if self.names is not None else f"V{idx + 1}"

    def loglik(self, data, *, separate: bool = False, as_pobs_data: bool = False):
        return dissmann_loglik(data, self, separate=separate, as_pobs_data=as_pobs_data)

    def pdf(self, data, *, as_pobs_data: bool = False):
        return np.exp(self.loglik(data, separate=True, as_pobs_data=as_pobs_data)["loglik"])

    def __repr__(self):
        return f"DissmannVine(dim={self.dim}, trees={len(self.edges)}, edges={self.n_edges})"


def _candidate_edges(nodes: list[_DissmannNode], tree: int):
    candidates = []
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            left = nodes[i]
            right = nodes[j]
            if tree == 1:
                shared = frozenset()
            else:
                parent_overlap = left.parent_ids & right.parent_ids
                if not parent_overlap:
                    continue
                shared = left.complete & right.complete
                if len(shared) != tree - 1:
                    continue
            left_diff = tuple(left.complete - shared)
            right_diff = tuple(right.complete - shared)
            if len(left_diff) != 1 or len(right_diff) != 1:
                continue
            x_var = left_diff[0]
            y_var = right_diff[0]
            x = left.cond_data.get((x_var, shared))
            y = right.cond_data.get((y_var, shared))
            if x is None or y is None:
                continue
            tau = empirical_kendall_tau(x, y)
            if not np.isfinite(tau):
                tau = 0.0
            candidates.append((abs(tau), tau, i, j, x_var, y_var, shared, x, y))
    return candidates


def _maximum_spanning_tree(candidates, n_nodes: int):
    parent = list(range(n_nodes))
    rank = [0] * n_nodes

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    selected = []
    for cand in sorted(candidates, key=lambda item: item[0], reverse=True):
        if union(cand[2], cand[3]):
            selected.append(cand)
            if len(selected) == n_nodes - 1:
                break
    if len(selected) != n_nodes - 1:
        raise ValueError("Dissmann candidate graph is disconnected; cannot build next vine tree")
    return selected


def _fit_candidate_pair(x, y, tau, familyset, selectioncrit, tau_threshold):
    if abs(tau) <= tau_threshold:
        fit = BiCop(0, 0.0)
    else:
        fit = bicop_select(x, y, familyset=familyset, selectioncrit=selectioncrit)
    loglik = bicop_loglik(x, y, fit, check_pars=False)
    criterion = -2.0 * loglik + (2.0 if selectioncrit.upper() == "AIC" else math.log(len(x))) * fit.npars
    return fit, float(loglik), float(criterion)


def dissmann_structure_select(
    data,
    *,
    familyset: list[int] | tuple[int, ...] | str | None = "all",
    selectioncrit: str = "AIC",
    trunc_lvl: int | None = None,
    n_trees: int | None = None,
    tau_threshold: float = 0.0,
    as_pobs_data: bool = True,
    names: list[str] | None = None,
    min_complete: int = 20,
) -> DissmannVine:
    """Select a regular vine by the sequential Dissmann maximum-spanning-tree algorithm.

    Parameters
    ----------
    familyset:
        Candidate bivariate families for `BiCopSelect`. Use `"all"` for every
        automatically selectable family, `"recommended"`/`"fast3"` for a fast
        three-family screen, `"gaussian"` for independence plus Gaussian, or an
        explicit list of family codes.
    n_trees:
        Number of vine tree layers to fit. Defaults to the full `d - 1` layers.
        `trunc_lvl` is accepted as a backward-compatible alias.
    """

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("data must be a two-dimensional matrix")
    if as_pobs_data:
        arr = pobs(arr)
    complete = np.all(np.isfinite(arr), axis=1)
    arr = arr[complete]
    if arr.shape[0] < min_complete:
        raise ValueError("not enough complete observations for structure selection")
    d = arr.shape[1]
    if names is not None and len(names) != d:
        raise ValueError("names must have one entry per variable")
    familyset = resolve_familyset(familyset)
    if n_trees is not None and trunc_lvl is not None and int(n_trees) != int(trunc_lvl):
        raise ValueError("n_trees and trunc_lvl are aliases; supply only one value")
    requested_trees = n_trees if n_trees is not None else trunc_lvl
    if requested_trees is not None and int(requested_trees) < 1:
        raise ValueError("n_trees must be at least 1")
    max_tree = d - 1 if requested_trees is None else min(int(requested_trees), d - 1)

    node_id = 0
    current_nodes: list[_DissmannNode] = []
    for var in range(d):
        current_nodes.append(
            _DissmannNode(
                node_id=node_id,
                complete=frozenset({var}),
                cond_data={(var, frozenset()): arr[:, var]},
            )
        )
        node_id += 1

    all_edges: list[list[DissmannEdge]] = []
    edge_id = 0
    for tree in range(1, max_tree + 1):
        candidates = _candidate_edges(current_nodes, tree)
        selected = _maximum_spanning_tree(candidates, len(current_nodes))
        next_nodes: list[_DissmannNode] = []
        tree_edges: list[DissmannEdge] = []
        for weight, tau, li, ri, x_var, y_var, shared, x, y in selected:
            left = current_nodes[li]
            right = current_nodes[ri]
            fit, loglik, criterion = _fit_candidate_pair(
                x, y, tau, familyset, selectioncrit, tau_threshold
            )
            left_given = np.asarray(bicop_hfunc2(x, y, fit, check_pars=False), dtype=float)
            right_given = np.asarray(bicop_hfunc1(x, y, fit, check_pars=False), dtype=float)
            left_given = np.clip(left_given, 1e-12, 1.0 - 1e-12)
            right_given = np.clip(right_given, 1e-12, 1.0 - 1e-12)
            left_cond = frozenset(set(shared) | {y_var})
            right_cond = frozenset(set(shared) | {x_var})
            new_node = _DissmannNode(
                node_id=node_id,
                complete=left.complete | right.complete,
                cond_data={
                    (x_var, left_cond): left_given,
                    (y_var, right_cond): right_given,
                },
                parent_ids=frozenset({left.node_id, right.node_id}),
            )
            next_nodes.append(new_node)
            tree_edges.append(
                DissmannEdge(
                    tree=tree,
                    edge_id=edge_id,
                    left_node=left.node_id,
                    right_node=right.node_id,
                    new_node=node_id,
                    conditioned=(x_var, y_var),
                    conditioning=tuple(sorted(shared)),
                    tau=float(tau),
                    weight=float(weight),
                    family=fit.family,
                    par=fit.par,
                    par2=fit.par2,
                    loglik=loglik,
                    criterion=criterion,
                )
            )
            node_id += 1
            edge_id += 1
        all_edges.append(tree_edges)
        current_nodes = next_nodes
        if len(current_nodes) <= 1:
            break

    return DissmannVine(
        dim=d,
        edges=all_edges,
        names=names,
        familyset=familyset,
        selectioncrit=selectioncrit.upper(),
        trunc_lvl=max_tree,
        n_trees=max_tree,
        tau_threshold=tau_threshold,
        nobs=arr.shape[0],
    )


def dissmann_loglik(data, model: DissmannVine, *, separate: bool = False, as_pobs_data: bool = False):
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("data must be a two-dimensional matrix")
    if arr.shape[1] != model.dim:
        raise ValueError("data dimension does not match model")
    if as_pobs_data:
        arr = pobs(arr)
    complete = np.all(np.isfinite(arr), axis=1)
    arr = arr[complete]
    n = arr.shape[0]
    nodes: dict[int, _DissmannNode] = {}
    for var in range(model.dim):
        nodes[var] = _DissmannNode(
            node_id=var,
            complete=frozenset({var}),
            cond_data={(var, frozenset()): arr[:, var]},
        )
    pointwise = np.zeros(n, dtype=float)
    values = {}
    for tree in model.edges:
        for edge in tree:
            left = nodes[edge.left_node]
            right = nodes[edge.right_node]
            x_var, y_var = edge.conditioned
            shared = frozenset(edge.conditioning)
            x = left.cond_data[(x_var, shared)]
            y = right.cond_data[(y_var, shared)]
            ll = np.asarray(
                bicop_loglik(x, y, edge.family, edge.par, edge.par2, separate=True, check_pars=False),
                dtype=float,
            )
            pointwise += ll
            values[edge.edge_id] = ll
            left_given = np.asarray(
                bicop_hfunc2(x, y, edge.family, edge.par, edge.par2, check_pars=False),
                dtype=float,
            )
            right_given = np.asarray(
                bicop_hfunc1(x, y, edge.family, edge.par, edge.par2, check_pars=False),
                dtype=float,
            )
            left_cond = frozenset(set(shared) | {y_var})
            right_cond = frozenset(set(shared) | {x_var})
            nodes[edge.new_node] = _DissmannNode(
                node_id=edge.new_node,
                complete=left.complete | right.complete,
                cond_data={
                    (x_var, left_cond): np.clip(left_given, 1e-12, 1.0 - 1e-12),
                    (y_var, right_cond): np.clip(right_given, 1e-12, 1.0 - 1e-12),
                },
                parent_ids=frozenset({left.node_id, right.node_id}),
            )
    return {"loglik": pointwise if separate else float(np.sum(pointwise)), "V": {"value": values}}


def RVineStructureSelect(data, familyset="all", **kwargs):
    return dissmann_structure_select(data, familyset=familyset, **kwargs)


def RVineCopSelect(data, familyset="all", Matrix=None, **kwargs):
    if Matrix is None:
        return dissmann_structure_select(data, familyset=familyset, **kwargs)
    selectioncrit = kwargs.pop("selectioncrit", "AIC")
    data = np.asarray(data, dtype=float)
    d = data.shape[1]
    fam = np.zeros((d, d), dtype=int)
    par = np.zeros((d, d), dtype=float)
    par2 = np.zeros((d, d), dtype=float)
    # A simple sequential first-tree selector. Higher trees are independence
    # unless supplied by a later specialized implementation.
    for j in range(d - 1):
        fit = bicop_select(data[:, j], data[:, j + 1], familyset=familyset, selectioncrit=selectioncrit)
        fam[d - 1, j] = fit.family
        par[d - 1, j] = fit.par
        par2[d - 1, j] = fit.par2
    return RVineMatrix(Matrix, fam, par, par2, check_pars=False)


RVineLogLik = rvine_loglik
RVinePDF = rvine_pdf
RVineAIC = rvine_aic
RVineBIC = rvine_bic
RVineSim = rvine_sim
RVinePar2Tau = rvine_par_to_tau
RVinePar2Beta = rvine_par_to_beta
RVineMatrixNormalize = RVineMatrixNormalize
RVineCopSelect = RVineCopSelect
RVineStructureSelect = RVineStructureSelect
