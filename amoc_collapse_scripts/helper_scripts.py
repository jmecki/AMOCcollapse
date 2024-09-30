import argparse
import ast
import contextlib
import json
import operator
import sys

import optim_esm_tools
import optim_esm_tools as oet

import os
import tempfile
import shutil
import numpy as np

import typing as ty

import matplotlib.pyplot as plt

import pandas as pd
import logging

import functools
import inspect
import itertools
import numba
import xarray as xr

_GLOBAL_WATER_MASK: dict = dict()


@numba.njit
def running_mean(a: np.ndarray, window: int) -> np.ndarray:
    res = np.zeros_like(a)
    res[:] = np.nan
    half_win = window // 2
    mean = 0
    for i, v in enumerate(a):
        mean += v
        if i >= window:
            mean -= a[i - window]
        if i >= (window - 1):
            res[i - half_win + 1] = mean / window
    return res


@numba.njit
def running_mean_array(a: np.ndarray, window: int) -> np.ndarray:
    _, len_x, len_y = a.shape
    res = np.zeros_like(a)
    res[:] = np.nan
    for i in range(len_x):
        for j in range(len_y):
            res[:, i, j] = running_mean(a[:, i, j], window)
    return res


def jump_n_years(field: str, ds_local: xr.Dataset, n_years: int = 10) -> np.float_:
    ma = int(oet.config.config["analyze"]["moving_average_years"])
    use_field = f"{field}_run_mean_{ma}"
    a = ds_local[use_field].mean("lat lon".split()).values

    return np.nanmax(np.abs(a[n_years:] - a[:-n_years]))


def get_water_mask_array(*a) -> np.ndarray:
    return get_water_mask(*a).values


def get_water_mask(water_mask_path: ty.Optional[str]) -> xr.DataArray:
    if "set" in _GLOBAL_WATER_MASK:
        set_water_da = _GLOBAL_WATER_MASK["set"]
        return set_water_da

    assert isinstance(water_mask_path, str)
    if not os.path.exists(water_mask_path):
        raise FileNotFoundError(f"No water_mask at {water_mask_path}")
    with tempfile.TemporaryDirectory() as temp_dir:
        shutil.copy2(
            water_mask_path,
            temp := os.path.join(temp_dir, "mask"),
        )
        water_mask_array: xr.DataArray = oet.load_glob(temp, load=True)["mask"].astype(
            np.bool_,
        )
    _GLOBAL_WATER_MASK["set"] = water_mask_array
    return water_mask_array


@functools.lru_cache(maxsize=int(1e9))
def cache_psym_test(values: np.ndarray, **kw) -> np.float64:
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    return oet.analyze.time_statistics.calculate_symmetry_test(values=values, **kw)


def _weighted_mean_array_xarray(
    data: xr.DataArray,
    weights: xr.DataArray,
) -> xr.DataArray:
    res = data * weights
    if "time" in res.dims:
        res = res.sum("lat lon".split())
        mask_time = data.isnull().all(dim="lat").all(dim="lon")
        res[mask_time] = np.nan

        mask_lat_lon = ~data.isnull().all(dim="time")
        area_mask = weights.where(mask_lat_lon)
    else:
        mask_lat_lon = ~data.isnull()
        res = res.values[mask_lat_lon].sum()
        area_mask = weights.where(mask_lat_lon)

    return res / (area_mask.sum())


def _weighted_mean_array_numpy(
    data: np.ndarray,
    weights: np.ndarray,
    has_time_dim: bool = True,
    _dtype=np.float64,
) -> np.ndarray:
    res = data * weights
    if has_time_dim:
        na_array = np.isnan(res)

        mask_time = na_array.all(axis=1).all(axis=1)

        # This is slightly confusing. We used to ignore any time step where there is a nan, but this is problematic if the nans pop in and out at a given grid cell
        # So instead we remove those grid cells.

        # Used to follow this logic: throw away all data where each cell is always nan in time, and keep data without any nan in time.
        # mask_lat_lon = ~na_array.all(axis=0)
        # no_na_vals = res[~mask_time][:, mask_lat_lon].sum(axis=1)

        # However, it's better to exclude those grid cells that are nan at least somewhere in time
        # Individual grid cells might have nans, but are at least not consistent in time.
        mask_lat_lon = ~na_array[~mask_time].any(axis=0)
        no_na_vals = np.nansum(res[~mask_time][:, mask_lat_lon], axis=1)

        res = np.zeros(len(data), dtype=_dtype)
        res[mask_time] = np.nan
        res[~mask_time] = no_na_vals
        area_mask = weights[mask_lat_lon]
    else:
        mask_lat_lon = ~np.isnan(data)
        res = np.nansum(res[mask_lat_lon])
        area_mask = weights[mask_lat_lon]

    return res / (area_mask.sum())


def _weighted_mean_array_numba(
    data: np.ndarray,
    weights: np.ndarray,
    has_time_dim: bool = True,
) -> ty.Union[float, np.ndarray]:
    if has_time_dim:
        assert len(data.shape) == 3, data.shape

    else:
        if len(data.shape) == 1:
            return _weighted_mean_1d_numba(data, weights)
        assert len(data.shape) == 2, data.shape
        return _weighted_mean_2d_numba(data, weights)
    return _weighted_mean_3d_numba(data, weights)


@numba.njit
def _weighted_mean_2d_numba(data, weights):
    tot = 0.0
    weight = 0.0
    x, y = data.shape
    for i in range(x):
        for j in range(y):
            if np.isnan(data[i][j]):
                continue
            tot += data[i][j] * weights[i][j]
            weight += weights[i][j]
    if tot == 0.0:
        return np.nan
    return tot / weight


@numba.njit
def _weighted_mean_1d_numba(data: np.ndarray, weights: np.ndarray) -> float:
    tot = 0.0
    weight = 0.0
    for i in range(len(data)):
        if np.isnan(data[i]):
            continue
        tot += data[i] * weights[i]
        weight += weights[i]
    if tot == 0.0:
        return np.nan
    return tot / weight


@numba.njit
def _weighted_mean_3d_numba(data, weights, _dtype=np.float64):
    t, x, y = data.shape
    is_nan_xy = np.zeros((x, y), dtype=np.bool_)
    weight = 0.0
    # Not sure if we should allow anything but np.float64 since you get overflows get quickly!
    tot = np.zeros(t, dtype=_dtype)

    # First, check which time steps are always nan
    for k in range(t):
        do_break = False
        for i in range(x):
            for j in range(y):
                if ~np.isnan(data[k][i][j]):
                    do_break = True
                    break
            if do_break:
                break
        is_nan_for_all_i_j = not do_break
        if is_nan_for_all_i_j:
            tot[k] = np.nan

    # Then, check which lat,lon coords are always nan
    for k in range(t):
        if np.isnan(tot[k]):
            continue
        for i in range(x):
            for j in range(y):
                if np.isnan(data[k][i][j]):
                    is_nan_xy[i][j] = True

    # Now sum all gridcells which are never nan in time, or lat+lon
    for i in range(x):
        for j in range(y):
            if is_nan_xy[i][j]:
                continue
            for k in range(t):
                if np.isnan(tot[k]):
                    continue
                tot[k] = tot[k] + data[k][i][j] * weights[i][j]
            weight += weights[i][j]
    return tot / weight


def weighted_mean_array(
    _ds: xr.Dataset,
    field: str = "std detrended",
    area_field: str = "cell_area",
    return_values: bool = True,
    method: str = "numpy",
    time_field: str = "time",
) -> ty.Union[np.ndarray, float, xr.DataArray]:
    if method == "xarray":
        res_da = _weighted_mean_array_xarray(_ds[field], _ds[area_field])
        return res_da.values if return_values else res_da

    da_sel = _ds[field]
    has_time_dim = time_field in da_sel.dims

    data = da_sel.values
    weights = _ds[area_field].values
    kw = dict(data=data, weights=weights, has_time_dim=has_time_dim)
    if method == "numba":
        res_arr = _weighted_mean_array_numba(**kw)  # type: ignore
    elif method == "numpy":
        res_arr = _weighted_mean_array_numpy(**kw)  # type: ignore
    else:
        raise ValueError(f"Unknown method {method}")

    if return_values or not has_time_dim:
        return res_arr
    res_da = xr.DataArray(res_arr, dims="time")
    res_da.attrs.update(da_sel.attrs)
    return res_da