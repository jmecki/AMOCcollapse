import collections
import datetime
import glob
import inspect
import itertools
import logging
import os
import shlex
import shutil
import string
import subprocess
import sys
import time
import typing as ty
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from functools import partial

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numba
import numpy as np
import optim_esm_tools as oet
import pandas as pd
import psutil
import regionmask
import scipy
import scipy.ndimage as ndimage
import xarray as xr
from immutabledict import immutabledict
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from tqdm.notebook import tqdm

from . import helper_scripts 

def _mask_xr_ds(
    data_set: xr.Dataset,
    masked_dims: ty.Iterable[str],
    ds_start: xr.Dataset,
    da_mask: xr.DataArray,
    keep_keys: ty.Optional[ty.Iterable[str]] = None,
):
    """Rebuild data_set for each variable that has all masked_dims."""
    for k, data_array in data_set.data_vars.items():
        if keep_keys is not None and k not in keep_keys:
            continue
        if all(dim in list(data_array.dims) for dim in masked_dims):
            da = data_set[k].where(da_mask, drop=False)
            da = da.assign_attrs(ds_start[k].attrs)
            data_set[k] = da

    return data_set


class BaseMaskedProducer:
    def __init__(self, ds: xr.Dataset, mask: xr.DataArray):
        if (
            isinstance(ds, xr.Dataset)
            and "lev" in ds.dims
            and "lev_size" not in ds.dims
        ):
            ds = ds.copy()
            ds["lev_size"] = xr.DataArray(np.diff(ds["lev_bnds"])[:, 0], dims="lev")

        self.ds = ds
        self.mask = mask

    def get_mask_from_kw(self):
        return self.mask

    def get_ds_masked(
        self,
        mask: xr.DataArray,
        ds: ty.Optional[xr.Dataset] = None,
    ) -> xr.Dataset:
        ds = ds or self.ds
        ds_sel = _mask_xr_ds(
            data_set=ds.copy(),
            masked_dims=["lat", "lon"],
            ds_start=ds,
            da_mask=mask,
        )
        return ds_sel

    def get_average_time_series(
        self,
        mask,
        time_field="time",
    ):
        ds_sel = self.ds
        if "lev" in ds_sel.dims:
            return weigthed_mean_irregular(
                ds_sel[ds_sel.variable_id],
                mask=mask,
                weights=ds_sel["lev_size"],
            )

        ds_sel = self.get_ds_masked(ds=ds_sel, mask=mask)
        return helper_scripts.weighted_mean_array(
            ds_sel,
            field=ds_sel.variable_id,
            method="numba",
            time_field=time_field,
        )

    def get_mean_time_series(
        self,
        time_to_year: bool = False,
        time_field="time",
        rm=None,
        **kw,
    ):
        if time_field not in self.ds:
            raise ValueError(time_field)
        for k, v in dict(label=self.ds.experiment_id).items():
            kw.setdefault(k, v)

        ds_sel = self.ds
        if time_to_year:
            times = np.array([t.year for t in ds_sel[time_field].values])
        else:
            times = ds_sel[time_field].values

        args_for_mask = inspect.getfullargspec(self.get_mask_from_kw)
        kw_for_mask = {k: v for k, v in kw.items() if k in args_for_mask.args}
        mask = self.get_mask_from_kw(**kw_for_mask)
        for k in list(kw_for_mask):
            kw.pop(k)

        arr_mean = self.get_average_time_series(
            mask=mask,
            time_field=time_field,
        )
        if rm is not None:
            arr_mean = helper_scripts.running_mean(arr_mean, rm)
        return times, arr_mean, kw

    def plot_mean_time_series(
        self,
        ax: ty.Optional[plt.Axes] = None,
        **kw,
    ):
        ax = ax or plt.gca()
        times, arr_mean, kw_remaining = self.get_mean_time_series(**kw)
        ax.plot(times, arr_mean, **kw_remaining)
        ax.set_ylabel(oet.plotting.plot.get_ylabel(self.ds))


class NorthAtlanticMaskedProducer(BaseMaskedProducer):
    def __init__(self, ds: xr.Dataset, mask: None = None):
        super().__init__(ds=ds, mask=mask)

    def get_mask_from_kw(self, hemisphere="north"):
        return self.get_amoc_mask(hemisphere=hemisphere)

    def get_amoc_mask(self, hemisphere: str, **kw) -> xr.DataArray:
        if hemisphere == "south":
            return self.get_amoc_mask_south(**kw)
        assert hemisphere == "north", f'{hemisphere} is not in ["north", "south"]'
        return self.get_amoc_mask_north(**kw)

    def get_amoc_mask_north(
        self,
        lat_min: float = 40.0,
        lat_max: float = 80.0,
        lon_min: float = -70.0,
        lon_max: float = 20.0,
    ) -> xr.DataArray:
        mask = self.read_water_mask()
        mask = self.approximate_atlantic(mask, lon_min=lon_min, lon_max=lon_max)
        mask &= mask.lat >= lat_min
        mask &= mask.lat <= lat_max
        return mask

    def get_amoc_mask_south(self) -> xr.DataArray:
        mask = self.read_water_mask()
        mask = self.approximate_atlantic(mask)
        mask &= mask.lat >= -50
        mask &= mask.lat <= -40
        mask &= (mask.lon < 20) | (mask.lon > 240)

        return mask

    @staticmethod
    def read_water_mask():
        return helper_scripts.get_water_mask("/data/volume_2/config_utils/is_water.nc")

    @staticmethod
    def approximate_atlantic(
        mask: xr.DataArray,
        lon_min: float = -70.0,
        lon_max: float = 20.0,
    ) -> xr.DataArray:
        new_mask = mask.copy()

        new_mask[:, ((new_mask.lon > 30) & (new_mask.lon < 285))] = False

        # 2d indexing on xr.DataArray doesn't work, so use numpy
        _new_mask_array = new_mask.values

        # everything outside 70W:20E
        _new_mask_array[
            (new_mask.lat <= 0)
            & (new_mask.lon < (360 + lon_min))
            & (new_mask.lon > lon_max)
        ] = False
        _new_mask_array[
            (new_mask.lat >= 0)
            & (new_mask.lon < (360 + lon_min))
            & (new_mask.lon > lon_max)
        ] = False

        # Remove mediteranian France/Italy
        _new_mask_array[
            (new_mask.lat > 10)
            & (new_mask.lat < 50)
            & (new_mask.lon < 290)
            & (new_mask.lon >= 0)
        ] = False

        # Remove Straight of Gibraltar
        _new_mask_array[
            (new_mask.lat > 10) & (new_mask.lat < 40) & (new_mask.lon >= 355)
        ] = False
        new_mask.data = _new_mask_array

        return new_mask

    def get_amoc_mask_north_old(self) -> xr.DataArray:
        ds = self.ds
        numbered_regions = regionmask.defined_regions.ar6.all.mask(ds.lon, ds.lat)
        north_atlantic_id = 50  # regionmask.defined_regions.ar6.all.names[50]
        mask = numbered_regions == north_atlantic_id
        mask &= mask.lat >= 35
        mask &= mask.lat <= 65
        return mask

    def get_amoc_mask_south_old(self) -> xr.DataArray:
        ds = self.ds
        numbered_regions = regionmask.defined_regions.ar6.all.mask(ds.lon, ds.lat)
        south_atlantic_id = 52  # regionmask.defined_regions.ar6.all.names[52]
        mask = numbered_regions == south_atlantic_id
        mask &= mask.lat >= -50
        mask &= mask.lat <= -40
        mask &= (mask.lon < 20) | (mask.lon > 240)
        return mask

    def plot_mean_time_series(self, *a, hemisphere="north", apply_amoc=False, **kw):
        if not apply_amoc:
            raise ValueError(
                "After changing the code, i'm not sure what this was supposed to do, maybe leave it true?",
            )
        super().plot_mean_time_series(*a, **kw, hemisphere=hemisphere)


class PlotProducer(NorthAtlanticMaskedProducer):
    ...


class MassProduce:
    def __init__(
        self,
        production_mapping: ty.Dict[str, PlotProducer],
        required_keys=ty.Tuple[str, ...],
    ) -> None:
        assert all(k in production_mapping for k in required_keys)
        self.production_mapping = production_mapping

    def plot_all(self) -> None:
        self.production_mapping["hdfs"].plot_mean_time_series(
            apply_amoc=True,
        )
        plt.show()

        self.production_mapping["wfo"].plot_mean_time_series(
            apply_amoc=True,
        )
        plt.show()


def yearly_average(ds):
    dds = ds.groupby("time.year").mean().rename(dict(year="time"))
    if "cell_area" in dds:
        del dds["cell_area"]
        dds["cell_area"] = ds["cell_area"].copy()
    return dds


def weigthed_mean_irregular(
    da: xr.DataArray,
    weights: xr.DataArray,
    mask: xr.DataArray,
    _incr=5,
):
    assert mask.dims == ("lat", "lon"), mask.dims
    assert da.dims == ("time", "lev", "lat", "lon"), da.dims
    assert weights.dims == ("lev",), weights.dims
    mask_array = mask.values.astype(bool)
    res = []
    for t in oet.utils.tqdm(np.arange(0, len(da["time"]) + _incr, _incr), disable=True):
        da_values = da.isel(time=slice(t, t + _incr)).load()
        values = da_values.values
        values[:, :, ~mask_array] = np.nan
        res += [_weighted_mean_time_irregular(values, weights.values)]
    return np.concatenate(res)


@numba.njit
def _weighted_mean_time_irregular(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    time, _, _, _ = data.shape
    res = np.zeros(time, dtype=np.float64)
    for t in range(time):
        res[t] = _weighted_mean_array_irregular(data[t], weights)
    return res


@numba.njit
def _weighted_mean_array_irregular(data: np.ndarray, weights: np.ndarray) -> float:
    d, lat, lon = data.shape
    assert len(weights) == d
    tot = 0.0
    tot_w = 0.0

    for z in range(d):
        for i in range(lat):
            for j in range(lon):
                if np.isnan(data[z][i][j]):
                    continue
                tot += data[z][i][j] * weights[z]
                tot_w += weights[z]
    if tot_w == 0.0:
        return np.nan
    return tot / tot_w


def set_time_int(ds: xr.Dataset) -> xr.Dataset:
    if not isinstance(ds["time"].values[0], (int, np.int_)):
        ds["time"] = [t.year for t in ds["time"].values]
    return ds




def update_units(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    if ds.attrs["variable_id"] in ["msftbarot", "sltovovrt"]:
        var = ds.attrs["variable_id"]
        ds[f"{var}"] = ds[var] / 1.0e9
        ds[f"{var}"].attrs["units"] = "Sv"
        if f"{var}_run_mean_10" in ds:
            ds[f"{var}_run_mean_10"] = ds[f"{var}_run_mean_10"] / 1.0e9
            ds[f"{var}_run_mean_10"].attrs["units"] = "Sv"
    if ds.attrs["variable_id"] == "tas":
        ds["tas"] = ds["tas"] - 272.15
        ds["tas"].attrs["units"] = "${}^\\circ$C"
        ds["tas_run_mean_10"] = ds["tas_run_mean_10"] - 272.15
        ds["tas_run_mean_10"].attrs["units"] = "${}^\\circ$C"
    return ds


def get_lev_coord(ds: xr.Dataset) -> str:
    for k in "lev olevel".split():
        if k in ds or k in ds.dims:
            return k
    raise ValueError


def scale_flux(ds):
    if ds[ds.variable_id].attrs.get("units") in ["kg m-2 s-1"]:
        _var = ds.variable_id
        oet.get_logger().warning(f"Update units for {_var}")
        for field in [_var, f"{_var}_run_mean_10"]:
            ds[field] = ds[field] * 1e5
            ds[field].attrs.update(dict(units=r"$10^{-5}$ kg\, m$^{-2}$\, s$^{-1}$"))


def get_vmin_vmax(da, minmax_percentile):
    ar = da.values.flatten()
    ar = ar[~np.isnan(ar)]
    if not len(ar):
        return dict()
    mm = max(
        np.abs(np.percentile(ar, minmax_percentile)),
        np.abs(np.percentile(ar, 100 - minmax_percentile)),
    )
    vmin = -mm if np.any(ar < 0) else 0
    vmax = mm if np.any(ar > 0) else 0
    return dict(vmin=vmin, vmax=vmax)


def plot_mlotst_difference(
    ds_glob: xr.Dataset,
    year_0: int = 1995,
    year_1: int = 2095,
    contours: bool = True,
    smooth: bool = True,
    mask: ty.Optional[xr.DataArray] = None,
    force_blue_red: bool = True,
    _crange_perc: ty.Union[float, int] = 1,
    **kw,
) -> None:
    if mask is None:
        raise ValueError
    da_reg = ds_glob[ds_glob.variable_id + "_run_mean_10"].where(mask, drop=True)
    da_t0 = da_reg.sel(time=year_0)
    da_t1 = da_reg.sel(time=year_1)
    da_filler = da_t1 - da_t0

    if force_blue_red:
        # mm = max(np.abs(float(da_filler.min())), np.abs(float(da_filler.max())))

        for key, value in dict(
            **get_vmin_vmax(da_filler, _crange_perc),
            cmap="coolwarm",
        ).items():
            kw.setdefault(key, value)
    plot_and_modify_buffer(da_filler, smooth, contours, **kw)
    plt.title(f"{year_1} - {year_0} {ds_glob.variable_id}")


def plot_da_2d(*a, **kw):
    oet.get_logger().warning("plot_da_2d has been replaced by plot_and_modify_buffer")
    return plot_and_modify_buffer(*a, **kw)


def plot_and_modify_buffer(
    da_filler: xr.DataArray,
    smooth: bool,
    contours: bool,
    minmax_percentile=None,
    **kw,
):
    if minmax_percentile:
        kw = kw.copy()
        mm = get_vmin_vmax(da_filler, minmax_percentile)
        if mm:
            kw.update(mm)
            if kw["vmin"] == -kw["vmax"]:
                kw.setdefault("cmap", "RdBu_r")
    plt.figure(dpi=300)
    oet.plotting.plot.setup_map(
        coastlines=False,
        projection="PlateCarree",
        gridline_kw=dict(draw_labels=True, linewidth=0.1),
    )
    plt.gca().coastlines(color="gray", linewidth=0.5)
    if smooth:
        vals = da_filler.values
        na_mask = np.isnan(vals)
        vals[na_mask] = 0
        sm_vals = ndimage.gaussian_filter(vals, sigma=(0.5, 0.5), order=0)
        sm_vals[na_mask] = np.nan
        if "vmin" in kw and "vmax" in kw:
            sm_vals = np.clip(sm_vals, kw["vmin"], kw["vmax"])
        da_filler.data = sm_vals
    da_filler.plot(
        transform=oet.plotting.plot.get_cartopy_transform("PlateCarree"),
        **kw,
    )
    if contours:
        # overlay_kw = dict()
        # if kw.get('vmin',0) < -1 and kw.get('vmax',0) > 1:
        #     overlay_kw = dict(levels=np.round(np.linspace(kw['vmin'], kw['vmax'], 10),0))
        overlay_contour(da_filler)

    plt.xlim(-70, 20)
    plt.ylim(40, 80)


def shifted_da(da):
    da_filler = da.copy()
    lon_v = da_filler["lon"].values

    lon_v[lon_v > 180] = lon_v[lon_v > 180] - 360
    # da_filler["lon"] = np.concatenate([np.arange(21), np.arange(-70, 0)])
    da_filler["lon"] = lon_v
    da_filler = da_filler.sortby("lon")
    return da_filler


def shift_da_back(da):
    da_filler = da.copy()

    lon_v = da_filler["lon"].values

    lon_v[lon_v < 0] = lon_v[lon_v < 0] + 360
    # da_filler["lon"] = np.concatenate([np.arange(360 - 70, 360), np.arange(21)])
    da_filler["lon"] = lon_v
    da_filler = da_filler.sortby("lon")
    return da_filler


def overlay_contour(da_filler: xr.DataArray, add_label=False, **kw):
    da_filler = shifted_da(da_filler)
    # CS = da_filler.plot.contour(levels=10, colors='k', transform=oet.plotting.plot.get_cartopy_transform('PlateCarree'))
    for k, v in dict(
        linewidths=0.5,
        colors="k",
        transform=oet.plotting.plot.get_cartopy_transform("PlateCarree"),
        levels=12,
    ).items():
        kw.setdefault(k, v)
    args = (da_filler.lon, da_filler.lat, da_filler.values)
    # kw.setdefault('levels', 12)

    # try:
    CS = plt.contour(*args, **kw)
    # except ValueError as e:
    #     oet.get_logger().info(f'Ran into {e} retrying with 15-levels')
    #     kw['levels'] = 15
    #     CS = plt.contour(*args, **kw)
    if add_label:
        plt.gca().clabel(CS, inline=True, fontsize=3)


def _base_mask(*a, lat_min=40, lat_max=80, lon_min=-75, lon_max=30, **kw):
    if a or kw:
        oet.get_logger().debug(f"Calling _base_mask doesn't require {a} or {kw}")
    return PlotProducer(None).get_amoc_mask_north(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


def _base_mask_old(ds, hemisphere=None):
    mask = ds["cell_area"].copy().astype(bool)
    mask.data[:] = False
    mask.data[10:50, -70:] = True
    mask.data[10:50, :21] = True
    if hemisphere is not None:
        mask = PlotProducer.approximate_atlantic(mask)

    return mask


def split_simple(values, mask, no_found_label=-1, n_clusters=2, **kw):
    """Thanks https://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html#sphx-glr-auto-examples-cluster-plot-segmentation-toy-py"""
    aa = values.copy()
    # if not np.sum(aa)>n_clusters:
    #     return aa
    aa[~mask] = 0
    aa[np.isnan(aa)] = 0
    sh = aa.shape
    if sh[0] < sh[1]:
        aa_square = np.zeros((sh[1], sh[1]), dtype=aa.dtype)
        aa_square[: sh[0]] = aa
    elif sh[1] > sh[0]:
        aa_square = np.zeros((sh[1], sh[1]), dtype=aa.dtype)
        aa_square[: sh[0]] = aa

    from sklearn.cluster import spectral_clustering
    from sklearn.feature_extraction import image

    assert no_found_label != 0, no_found_label
    img = aa_square
    mask = img.astype(bool)

    graph = image.img_to_graph(img, mask=mask)
    kw.setdefault("eigen_solver", "arpack")
    labels = spectral_clustering(graph, n_clusters=n_clusters, **kw)

    label_im = np.full(mask.shape, no_found_label)
    label_im[mask] = labels
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # axs[0].matshow(img)
    # axs[1].matshow(label_im)
    # plt.title(n_clusters)
    # plt.show()
    return label_im[: sh[0], : sh[1]]


# def _manual_resolve():
#     masked_vals = da.copy().values
#     masked_vals[sets!=id_key] = np.nan
#     mean_func_of_lon = np.nanmean(masked_vals, axis=1)
#     plt.imshow(masked_vals)
#     plt.show()
#     na_means = np.all(np.isnan(masked_vals), axis=1)
#     print(dict(mean_func_of_lon=mean_func_of_lon, na_means=na_means))
#     assert len(mean_func_of_lon) == len(masked_vals), (len(mean_func_of_lon) , len(masked_vals))
#     # Do not break in the first or last 5% of the cells
#     non_breakable = len(mean_func_of_lon[~na_means]) // 20
#     idx_lon_break = np.argmin(mean_func_of_lon[~na_means][non_breakable:-non_breakable])
#     true_idx_break_accounting_for_nan = np.argwhere(mean_func_of_lon==mean_func_of_lon[~na_means][idx_lon_break])[0,0]
#     print(dict(
#         idx_lon_break=idx_lon_break,
#         val = mean_func_of_lon[~na_means][idx_lon_break],
#         true_idx_break_accounting_for_nan=true_idx_break_accounting_for_nan,
#         val2=mean_func_of_lon[true_idx_break_accounting_for_nan]
#         ))
#     ma_group.data[(sets == id_key)][idx_lon_break:] = np.unique(ma_group.values)[-1] + 1


def _resolve_group_min_max(
    label_mask,
    mlotst_values,
    min_cells,
    max_cells,
    no_group_value,
):
    """Keep iterating label-mask to ensure each mask has > min_cels and <
    max_cells."""
    something_split = True
    something_discarded = True
    while something_discarded or something_split:
        something_split = False
        something_discarded = False
        for id_key, counts in zip(*np.unique(label_mask.values, return_counts=True)):
            if id_key == no_group_value:
                continue
            if max_cells and counts > max_cells:
                something_split = True
                _da_buffer = mlotst_values.copy()
                _da_buffer.data = label_mask.values == id_key
                _da_buffer = _da_buffer.astype(bool)
                n_cluster = int(max(np.round(counts / max_cells, 0), 2))
                while True:
                    new_groups = split_simple(
                        shifted_da(mlotst_values).values,
                        shifted_da(_da_buffer),
                        n_clusters=n_cluster,
                        no_found_label=no_group_value,
                    )
                    if all(
                        (i == no_group_value or (new_groups == i).sum() < max_cells)
                        for i in np.unique(new_groups)
                    ):
                        break

                    n_cluster += 1
                _da_buffer = _da_buffer.astype(np.int64)
                _da_buffer.data = new_groups

                new_groups = shift_da_back(_da_buffer).values
                label_mask.data[label_mask.values == id_key] = no_group_value
                for new_id in np.unique(new_groups):
                    if new_id == no_group_value:
                        continue
                    label_mask.data[(new_groups == new_id)] = max(
                        np.unique(label_mask.values)[-1] + 1,
                        1,
                    )
            if counts < min_cells:
                something_discarded = True
                label_mask.data[label_mask.values == id_key] = no_group_value
    return label_mask


# def _resolve_group_min_max(label_mask, mlotst_values, min_cells, max_cells, no_group_value):
#     """Keep iterating label-mask to ensure each mask has > min_cels and < max_cells"""
#     something_split=True
#     something_discarded=True
#     while something_discarded or something_split:
#         something_split=False
#         something_discarded=False
#         for id_key, counts in zip(*np.unique(label_mask.values, return_counts=True)):
#             if id_key == no_group_value:
#                 continue
#             if max_cells and counts > max_cells:
#                 something_split=True
#                 _da_buffer = mlotst_values.copy()
#                 _da_buffer.data = label_mask.values==id_key
#                 _da_buffer = _da_buffer.astype(bool)
#                 new_groups = split_simple(shifted_da(mlotst_values).values, shifted_da(_da_buffer), n_clusters=int(max(np.round(counts/max_cells, 0), 2)), no_found_label=no_group_value)
#                 _da_buffer = _da_buffer.astype(np.int64)
#                 _da_buffer.data = new_groups

#                 new_groups=shift_da_back(_da_buffer).values
#                 label_mask.data[label_mask.values==id_key] = no_group_value
#                 for new_id in np.unique(new_groups):
#                     if new_id == no_group_value:
#                         continue
#                     label_mask.data[(new_groups == new_id)] = max(np.unique(label_mask.values)[-1] + 1, 1)
#             if counts < min_cells:
#                 something_discarded=True
#                 label_mask.data[label_mask.values == id_key] = no_group_value
#     return label_mask


@dataclass
class MlotstRegion:
    label: str
    labels: ty.List[str]
    median_coords: ty.List[ty.Tuple[float, float]]
    masks: ty.List[np.ndarray]
    # averaged_values : ty.List[np.ndarray]
    timestamps: np.ndarray
    region_map: xr.DataArray
    da_basis: xr.DataArray
    full_ds: xr.Dataset
    year_str: str
    region_names: ty.Dict[str, str]
    da_global: xr.DataArray

    @property
    def averaged_values(self) -> ty.List[np.ndarray]:
        return [
            helper_scripts.weighted_mean_array(
                self.full_ds.where(mask, drop=False),
                field=self.full_ds.variable_id,
            )
            for mask in self.masks
        ]

    @property
    def region_dict(self):
        return {k: v for k, v in zip(self.labels, self.masks)}


def _split_grouped_cells_for_named_region(
    ds,
    ma_group,
    no_group_value,
    rename_dict=None,
    database=None,
    min_cells=None,
):
    database = (
        regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50 or database
    )
    min_cells = min_cells or -np.inf
    region_values = database.mask(ds).where(_base_mask(), drop=True)
    region_values_np = region_values.values
    region_names = database.names
    rename_dict = rename_dict or dict()

    # maybe we end up splitting two groups in one named region, not sure if that is bad?
    split_groups = dict()
    for group_id in np.unique(ma_group):
        if group_id == no_group_value:
            continue
        group_mask = ma_group == group_id

        this_region = region_values_np.copy()
        this_region[~group_mask] = no_group_value

        for region_id in np.unique(this_region):
            if np.isnan(region_id):
                continue
            region_id = region_id.astype(np.int64)
            if region_id == no_group_value:
                continue
            try:
                this_region_name = region_names[region_id]
            except IndexError:
                print(region_names, region_id)
            this_region_name = rename_dict.get(this_region_name, this_region_name)
            this_region_bool = this_region == region_id
            if this_region_bool.sum() < min_cells:
                continue
            if this_region_name in split_groups:
                split_groups[this_region_name] |= this_region_bool
            else:
                split_groups[this_region_name] = this_region_bool
            print(this_region_name, split_groups[this_region_name].sum())
    return split_groups


def get_names_of_mask(mask: xr.DataArray, rename_dict=None, database=None):
    rename_dict = rename_dict or dict()
    database = (
        database or regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50
    )
    mask_arr = database.mask(mask)
    names = []
    for i, n in enumerate(database.names):
        if np.sum((mask_arr == i) & (mask)):
            names.append(rename_dict.get(n, n))
    return ", ".join(sorted(set(names)))


def plot_mlotst_cells(*a, **kw):
    res = plot_mlotst_cells_full_ret(*a, **kw)
    return res.region_dict


@oet.utils.check_accepts(
    accepts=dict(split_by=["named_regions", "clustering"], mean_or_max=["mean", "max"]),
)
def plot_mlotst_cells_full_ret(
    ds,
    year_sel=1995,
    field=None,
    label=None,
    reference_depth=300,
    save_kw=None,
    max_cells=None,
    show=False,
    smooth_reference=True,
    min_cells=10,
    mean_or_max: str = "mean",
    split_by="named_regions",
    _split_kw=None,
    rename_dict=None,
) -> MlotstRegion:
    field = field or ds.variable_id + "_run_mean_10"
    rename_dict = rename_dict or dict()
    ds_sel = ds.copy()
    set_time_int(ds_sel)
    # ma = PlotProducer(ds).get_amoc_mask_north()
    # ds_sel = ds_sel.sel(time=year_sel)[field].load().where(ma, drop=False)

    mask = _base_mask(ds)

    ds_sel = ds.copy()
    set_time_int(ds_sel)
    if isinstance(year_sel, int):
        da_global = ds_sel.sel(time=year_sel)[field].load()
        da = da_global.where(mask, drop=True)
        year_str = str(year_sel)
    elif isinstance(year_sel, slice):
        from .manual_msft import slice_to_string

        if mean_or_max == "max":
            da = (
                ds_sel.sel(time=year_sel)[field]
                .max("time")
                .load()
                .where(mask, drop=True)
            )
        elif mean_or_max == "mean":
            da_global = ds_sel.sel(time=year_sel)[field].mean("time").load()
            da = da_global.where(mask, drop=True)
        else:
            raise ValueError
        year_str = f"{mean_or_max} in " + slice_to_string(year_sel).replace("$", "")

    plot_and_modify_buffer(
        da.copy(),
        smooth=True,
        contours=True,
        cbar_kwargs=dict(orientation="horizontal"),
    )
    plt.title(f"{label} in {year_str}")
    if save_kw is not None:
        kw = save_kw.copy()
        kw["name"] = kw["name"] + f"_in_{year_str}"
        oet.utils.save_fig(**kw)
    oet.plotting.plot._show(show)

    plt.figure(figsize=(12, 6))
    # da = ds_sel.sel(time=year_sel)[field].load().where(mask, drop=True)

    if smooth_reference:
        da_filler = da.copy()
        vals = da_filler.values
        na_mask = np.isnan(vals)
        vals[na_mask] = 0

        sm_vals = ndimage.gaussian_filter(vals, sigma=(1, 1), order=0)
        sm_vals[na_mask] = np.nan
        da_filler.data = sm_vals
        m_all = da_filler > reference_depth
    else:
        m_all = da > reference_depth
    ma_group = m_all.astype(np.int16)
    no_group_value = -1
    sets = oet.analyze.clustering.masks_array_to_coninuous_sets(
        [ma_group.values],
        no_group_value=no_group_value,
    )[0]
    ma_group.data = sets

    if split_by == "clustering":
        ma_group = _resolve_group_min_max(
            ma_group,
            mlotst_values=da if not smooth_reference else da_filler,
            min_cells=min_cells,
            max_cells=max_cells,
            no_group_value=no_group_value,
        )

        single_sets = []
        for idx in np.unique(ma_group.values):
            if idx == no_group_value:
                continue
            single_sets.append(ma_group.values == idx)
    elif split_by == "named_regions":
        _split_kw = _split_kw or dict()
        groups = _split_grouped_cells_for_named_region(
            ds,
            ma_group.copy(),
            no_group_value,
            rename_dict=rename_dict,
            **_split_kw,
        )
        single_sets = []
        ma_group.data[:] = no_group_value
        for i, g in enumerate(list(groups.values())):
            if np.sum(g) < min_cells:
                continue
            ma_group.data[g] = i + 1
            g_set = g.astype(bool)
            single_sets.append(g_set)

    # single_sets = oet.analyze.clustering._split_to_continous(
    #     [ma_group.values.astype(np.bool_)],
    # )
    ma_group.data[ma_group == no_group_value] = 0
    plot_and_modify_buffer(
        ma_group,
        smooth=False,
        contours=False,
        add_colorbar=False,
        cmap="Blues",
    )

    single_sets = sorted(
        single_sets,
        key=lambda x: partial(median_lat_lon, ds=ds)(x)[1],
    )
    masks = dict()
    region_names = dict()
    for i, ss in enumerate(single_sets):
        s = m_all.copy()
        s.data = ss

        k = string.ascii_uppercase[i]
        y, x = median_lat_lon(ss, ds)
        plt.text(x, y, k, c="r", ha="center", va="center")
        masks[k] = s.copy()
        region_names[k] = get_names_of_mask(s, rename_dict=rename_dict)

    plt.title(f"Regions {label} $>$ {reference_depth} m in {year_str}")
    if save_kw is not None:
        kw = save_kw.copy()
        kw["name"] = kw["name"] + "_regions"
        oet.utils.save_fig(**kw)
    oet.plotting.plot._show(show)

    averaged_values = []
    for k, s in masks.items():
        ds_reg = ds_sel.where(s, drop=False)
        y = helper_scripts.weighted_mean_array(ds_reg, field=field)
        plt.plot(ds_reg["time"], y, label=f"Reg. {k} of {label}")
        averaged_values.append(y)
    plt.ylabel("mlotst (rm$_{10}$) [m]" if field == "mlotst_run_mean_10" else field)
    if masks:
        plt.legend(**oet.utils.legend_kw(ncol=1))
    plt.ylim(0, None)
    if save_kw is not None:
        kw = save_kw.copy()
        kw["name"] = kw["name"] + "_region_time_series"
        oet.utils.save_fig(**kw)
    oet.plotting.plot._show(show)

    result = MlotstRegion(
        label=label,
        labels=list(masks),
        masks=list(masks.values()),
        median_coords=[median_lat_lon(ss, ds) for ss in single_sets],
        #   averaged_values=averaged_values,
        timestamps=ds_reg["time"].values if averaged_values else None,
        region_map=ma_group,
        da_basis=da,
        full_ds=ds.copy(),
        year_str=year_str,
        region_names=region_names,
        da_global=da_global,
    )

    return result


def median_lat_lon(m, ds):
    mask = ds["cell_area"].copy().astype(bool)
    mask = mask.where(_base_mask(), drop=True)
    lat, lon = np.meshgrid(mask.lat, mask.lon)
    x = np.median(np.mod(lon.T[m] - 180, 360)) - 180
    y = np.median(lat.T[m])
    return y, x


def mask_to_full_mask(mask, fill_value=None):
    mask_full = _base_mask().copy()
    mask_full.data = np.arange(np.product(mask_full.shape), dtype=np.int64).reshape(
        mask_full.shape,
    )
    keep_id = mask_full.where(_base_mask(), drop=True)
    bool_mask = mask.values.astype(np.bool_)
    if np.issubdtype(mask.values.dtype, np.floating):
        bool_mask &= ~np.isnan(mask.values)
        fill_value = fill_value if fill_value is not None else np.nan
    else:
        fill_value = fill_value if fill_value is not None else 0
    keep_id.data[~bool_mask] = np.nan

    all_val = mask_full.values.flatten().astype(mask.dtype)
    all_val[:] = fill_value
    for x, v in zip(keep_id.values.flatten(), mask.values.flatten()):
        if not np.isnan(x):
            all_val[int(x)] = v

    mask_full.data = all_val.reshape(mask_full.shape)
    return mask_full
