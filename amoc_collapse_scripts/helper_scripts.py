import argparse
import ast
import contextlib
import functools
import inspect
import itertools
import json
import logging
import operator
import os
import shutil
import sys
import tempfile
import typing as ty

import matplotlib.pyplot as plt
import numba
import numpy as np
import optim_esm_tools
import optim_esm_tools as oet
import pandas as pd
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


def weighted_mean_array(
    _ds: xr.Dataset,
    field: str = "std detrended",
    area_field: str = "cell_area",
    return_values: bool = True,
    method: str = "numpy",
    time_field: str = "time",
) -> ty.Union[np.ndarray, float, xr.DataArray]:
    da_sel = _ds[field]
    has_time_dim = time_field in da_sel.dims

    data = da_sel.values
    weights = _ds[area_field].values
    kw = dict(data=data, weights=weights, has_time_dim=has_time_dim)
    if method == "numpy":
        res_arr = _weighted_mean_array_numpy(**kw)  # type: ignore
    else:
        raise ValueError(f"Unknown method {method}")

    if return_values or not has_time_dim:
        return res_arr
    res_da = xr.DataArray(res_arr, dims="time")
    res_da.attrs.update(da_sel.attrs)
    return res_da
