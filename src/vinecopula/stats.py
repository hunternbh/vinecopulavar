from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ._utils import empirical_kendall_tau, rankdata_average


@dataclass
class FredMDData:
    data: np.ndarray
    names: list[str]
    dates: list[str]
    transform_codes: dict[str, int]
    raw: np.ndarray | None = None


def pobs(x, *, lower_tail: bool = True, ties_method: str = "average"):
    arr = np.asarray(x, dtype=float)
    if ties_method != "average":
        raise NotImplementedError("Only average ties are implemented in the Python port")
    if arr.ndim == 1:
        mask = np.isfinite(arr)
        out = np.full(arr.shape, np.nan, dtype=float)
        out[mask] = rankdata_average(arr[mask]) / (np.sum(mask) + 1.0)
    elif arr.ndim == 2:
        out = np.full(arr.shape, np.nan, dtype=float)
        for j in range(arr.shape[1]):
            mask = np.isfinite(arr[:, j])
            out[mask, j] = rankdata_average(arr[mask, j]) / (np.sum(mask) + 1.0)
    else:
        raise ValueError("x must be a vector or a two-dimensional array")
    return out if lower_tail else 1.0 - out


def EmpCDF(x):
    x = np.asarray(x, dtype=float)
    x = np.sort(x[np.isfinite(x)])
    n = len(x)
    if n == 0:
        raise ValueError("x must contain at least one finite value")

    def cdf(xx):
        xx_arr = np.asarray(xx, dtype=float)
        counts = np.searchsorted(x, xx_arr, side="right")
        out = np.maximum(counts, 1) / (n + 1.0)
        return out.item() if out.shape == () else out

    return cdf


def TauMatrix(data):
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a matrix")
    d = data.shape[1]
    out = np.eye(d)
    for i in range(d):
        for j in range(i):
            out[i, j] = out[j, i] = empirical_kendall_tau(data[:, i], data[:, j])
    return out


def as_copuladata(x):
    return np.asarray(x, dtype=float)


def pairs_copuladata(x, *_, **__):
    raise NotImplementedError("pairs.copuladata is an R plotting helper and is not ported")


def _parse_float(value: str) -> float:
    value = value.strip()
    if value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _fred_md_transform_column(values: np.ndarray, code: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = x.copy()
    if code == 1:
        return out
    if code == 2:
        out[1:] = np.diff(x)
        out[0] = np.nan
        return out
    if code == 3:
        out[2:] = np.diff(x, n=2)
        out[:2] = np.nan
        return out
    if code == 4:
        out = np.where(x > 0, np.log(x), np.nan)
        return out
    if code == 5:
        lx = np.where(x > 0, np.log(x), np.nan)
        out[1:] = np.diff(lx)
        out[0] = np.nan
        return out
    if code == 6:
        lx = np.where(x > 0, np.log(x), np.nan)
        out[2:] = np.diff(lx, n=2)
        out[:2] = np.nan
        return out
    if code == 7:
        growth = np.full_like(x, np.nan, dtype=float)
        growth[1:] = x[1:] / x[:-1] - 1.0
        out[1:] = np.diff(growth)
        out[0] = np.nan
        return out
    return out


def _parse_date_for_filter(value: str):
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def load_fred_md_csv(
    path,
    *,
    apply_transforms: bool = True,
    as_pobs_data: bool = True,
    max_missing: float = 0.2,
    fill: str | None = None,
    start_date: str | None = None,
) -> FredMDData:
    """Load a FRED-MD CSV with its ``Transform:`` row.

    Parameters mirror common FRED-MD workflows: transformation codes are read
    from the second row, columns with too much missing data are dropped, and the
    remaining complete rows can be converted to pseudo-observations for copula
    modeling.
    """

    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        first = next(reader)
        if first[0].strip() != "Transform:":
            raise ValueError("Expected FRED-MD Transform: row as the first data row")
        names = header[1:]
        transform_codes = {name: int(float(code)) for name, code in zip(names, first[1:]) if code.strip()}
        dates: list[str] = []
        rows: list[list[float]] = []
        for row in reader:
            if not row:
                continue
            date = row[0].strip()
            dates.append(date)
            rows.append([_parse_float(value) for value in row[1:]])

    raw = np.asarray(rows, dtype=float)
    if start_date is not None:
        cutoff = _parse_date_for_filter(start_date)
        if cutoff is None:
            raise ValueError("start_date must look like 'YYYY-MM-DD' or 'M/D/YYYY'")
        row_keep = np.array([
            (parsed is not None and parsed >= cutoff)
            for parsed in (_parse_date_for_filter(date) for date in dates)
        ])
        raw = raw[row_keep]
        dates = [date for date, keep in zip(dates, row_keep) if keep]
    data = raw.copy()
    if apply_transforms:
        for j, name in enumerate(names):
            data[:, j] = _fred_md_transform_column(data[:, j], transform_codes.get(name, 1))

    missing_rate = np.mean(~np.isfinite(data), axis=0)
    keep_cols = missing_rate <= max_missing
    data = data[:, keep_cols]
    raw = raw[:, keep_cols]
    names = [name for name, keep in zip(names, keep_cols) if keep]
    transform_codes = {name: transform_codes[name] for name in names if name in transform_codes}

    if fill is not None:
        if fill not in {"ffill", "bfill", "ffill_bfill"}:
            raise ValueError("fill must be one of None, 'ffill', 'bfill', or 'ffill_bfill'")
        if fill in {"ffill", "ffill_bfill"}:
            for j in range(data.shape[1]):
                last = np.nan
                for i in range(data.shape[0]):
                    if np.isfinite(data[i, j]):
                        last = data[i, j]
                    elif np.isfinite(last):
                        data[i, j] = last
        if fill in {"bfill", "ffill_bfill"}:
            for j in range(data.shape[1]):
                last = np.nan
                for i in range(data.shape[0] - 1, -1, -1):
                    if np.isfinite(data[i, j]):
                        last = data[i, j]
                    elif np.isfinite(last):
                        data[i, j] = last

    keep_rows = np.all(np.isfinite(data), axis=1)
    data = data[keep_rows]
    raw = raw[keep_rows]
    dates = [date for date, keep in zip(dates, keep_rows) if keep]
    if as_pobs_data:
        data = pobs(data)
    return FredMDData(data=data, names=names, dates=dates, transform_codes=transform_codes, raw=raw)


as_copuladata.__name__ = "as.copuladata"
pairs_copuladata.__name__ = "pairs.copuladata"
