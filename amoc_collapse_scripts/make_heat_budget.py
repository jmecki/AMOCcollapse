import collections
import datetime
import glob
import inspect
import itertools
import json
import logging
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import typing as ty
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from functools import partial

import amoc_deep_dive
import amoc_deep_dive_group
import cartopy.crs as ccrs
import manual_msft
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numba
import numpy as np
import optim_esm_tools as oet
import pandas as pd
import psutil
import regionmask
import scipy.ndimage
import seawater
import xarray as xr
from immutabledict import immutabledict
from IPython.display import display
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from scipy.signal.windows import gaussian
from sklearn.cluster import HDBSCAN
from tqdm.notebook import tqdm

rename_dict = {
    "Greenland Sea": "Nordic Seas",
    "Norwegian Sea": "Nordic Seas",
    "North Atlantic Ocean": "Irminger Sea",
    "Baffin Bay": "Labrador Sea",
    "Davis Strait": "Labrador Sea",
}


def smoother_lowess_year(a, n_year=40, **kw):
    frac = n_year / len(a)
    if "frac" in kw:
        raise ValueError
    return partial(
        oet.analyze.tools.smooth_lowess,
        **{
            "it": 0,
            "delta": 0.0,
            "xvals": None,
            "is_sorted": False,
            "missing": "drop",
            "return_sorted": True,
        },
    )(a, frac=frac)


smoother = smoother_lowess_year


def get_res_bucket(label, folder_dict):
    result_bucket = {}
    for ll in sorted(folder_dict):
        if ll != label:
            continue
        print(f"------------{ll}--------------")
        ds_mlotst = oet.read_ds(
            folder_dict[ll]["folder_mlotst"],
            add_history=True,
            max_time=None,
        ).load()
        amoc_deep_dive.set_time_int(ds_mlotst)

        result_bucket[ll] = amoc_deep_dive.plot_mlotst_cells_full_ret(
            ds_mlotst,
            reference_depth=650,
            smooth_reference=True,
            min_cells=25,
            max_cells=200,
            show=False,
            label=label,
            field="mlotst_march",
            year_sel=slice(1965, 1995),
            mean_or_max="mean",
            _split_kw=dict(min_cells=5),
            rename_dict=rename_dict,
        )
    return result_bucket


def atlantic_reference_mask(ds_reference):
    keep_idx = [0, 2, 14, 17, 20, 31, 32, 35, 39, 40, 41, 55, 56, 57, 60, 65, 81, 83]
    tot = None
    reg = regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50.mask_3D(
        ds_reference,
    )
    tot = reg.sel(region=keep_idx)
    tot = tot.sum("region")

    tot = tot.where(tot.lat > 26.5, drop=False)
    tot.data[tot.isnull()] = 0
    tot.astype(bool)
    return tot


@numba.njit
def _weighted_product_time_irregular(
    data: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    time, _, _, _ = data.shape
    res = np.zeros(time, dtype=np.float64)
    for t in range(time):
        res[t] = _weighted_product_array_irregular(data[t], weights)
    return res


@numba.njit
def _weighted_product_array_irregular(
    data: np.ndarray,
    weights: np.ndarray,
) -> np.float64:
    d, lat, lon = data.shape
    assert weights.shape == data.shape
    tot: np.float64 = 0.0

    for z in range(d):
        for i in range(lat):
            for j in range(lon):
                if np.isnan(data[z][i][j]):
                    continue
                if np.isnan(weights[z][i][j]):
                    continue
                tot += data[z][i][j] * weights[z][i][j]

    return tot


def weigthed_product_irregular(
    da: xr.DataArray,
    weights: xr.DataArray,
    mask: xr.DataArray,
    _incr=5,
    tqdm=True,
):
    assert mask.dims == ("lat", "lon"), mask.dims
    assert da.dims == ("time", "lev", "lat", "lon"), da.dims
    assert weights.dims == ("lev", "lat", "lon"), weights.dims
    mask_array = mask.values.astype(bool)
    res = []
    for t in oet.utils.tqdm(
        np.arange(0, len(da["time"]) + _incr, _incr),
        disable=not tqdm,
        desc="weigthed_product_irregular",
    ):
        da_values = da.isel(time=slice(t, t + _incr)).load()
        values = da_values.values
        values[:, :, ~mask_array] = np.nan
        res += [_weighted_product_time_irregular(values, weights.values)]
    return np.concatenate(res)


def build_thkcello(ds_depth, ds_thetao):
    weigths_array = (
        ds_thetao["lev_size"].copy() * xr.ones_like(ds_thetao["cell_area"]).copy()
    )

    wa = weigths_array.values
    for i, l in enumerate(wa):
        weight_sum = wa[: i + 1].sum(axis=0)
        overflow = weight_sum - ds_depth["deptho"].values
        wa[i][overflow > 0] = (l - overflow)[overflow > 0]

    wa[:, np.isnan(ds_depth["deptho"].values)] = np.nan
    weigths_array.data = wa
    weigths_array = weigths_array * ds_thetao["cell_area"]
    return weigths_array


def plot_pw_new(
    path_thetao,
    mask_north_atlantic,
    mask_north_atlantic_n45,
    add_label=None,
):
    ds_thetao = oet.load_glob(path_thetao)
    amoc_deep_dive.set_time_int(ds_thetao)
    lev_name = manual_msft.get_lev_coord(ds_thetao)
    if lev_name != "lev":
        print(ds_thetao[ds_thetao.variable_id].dims)
        ds_thetao = ds_thetao.rename(
            **{lev_name: "lev", f"{lev_name}_bnds": "lev_bnds"},
        )
        print(ds_thetao[ds_thetao.variable_id].dims)
    ds_thetao["lev_size"] = xr.DataArray(
        np.diff(ds_thetao["lev_bnds"])[:, 0],
        dims="lev",
    )
    p = glob.glob(f"/data/volume_2/so_thetao_amoc/thk/*{ds_thetao.source_id}*")[0]
    dest = p.replace("thk/", "thk/regrid/")
    if not os.path.exists(dest):
        oet.analyze.pre_process._remove_bad_vars(p)
        import cdo

        cdo_int = cdo.Cdo()
        cdo_int.remapbil("n90", input=p, output=dest)
    ds_depth = oet.load_glob(dest)
    if "time" in ds_depth[ds_depth.variable_id].dims:
        ds_depth = ds_depth.isel(time=0)

    def calculate_heating_power_pw(dTdxdydz, dt_year=1):
        rho = 1025  # kg/m3
        c = 4000  # J/(kg C)
        dt = dt_year * 365 * 24 * 3600  # s
        return c * rho * (dTdxdydz) / dt / 1e15

    if ds_depth.variable_id != "thkcello":
        print("Manually calculating thk-cello!")
        da_depth = build_thkcello(ds_depth, ds_thetao)
    else:
        lev_name = manual_msft.get_lev_coord(ds_depth)
        ds_depth = ds_depth.rename(**{lev_name: "lev"})
        da_depth = ds_depth["thkcello"].transpose(*"lev lat lon".split())
        da_depth = da_depth * ds_thetao["cell_area"]

    res_thetao = weigthed_product_irregular(
        ds_thetao["thetao"],
        weights=da_depth,
        mask=mask_north_atlantic,
    )
    res_thetao_n45 = weigthed_product_irregular(
        ds_thetao["thetao"],
        weights=da_depth,
        mask=mask_north_atlantic_n45,
    )

    time = ds_thetao["time"].values[1:]
    values_26 = calculate_heating_power_pw(np.diff(res_thetao))
    values_45 = calculate_heating_power_pw(np.diff(res_thetao_n45))
    plt.plot(time, smoother(values_26), label=rf"Ocean heating $>26^\circ$N{add_label}")
    plt.plot(time, smoother(values_45), label=rf"Ocean heating $>45^\circ$N{add_label}")

    return time, values_26, values_45


def plot_pw_old(
    path_thetao,
    mask_north_atlantic,
    mask_north_atlantic_n45,
    add_label=None,
):
    add_label = add_label if add_label is not None else ""

    def calculate_heating_power_pw(dT, mask, volume, dt_year=1):
        rho = 1025  # kg/m3
        c = 4000  # J/(kg C)
        dt = dt_year * 365 * 24 * 3600  # s

        volume = float(np.sum(volume.where(mask)))
        mass = rho * volume
        return c * mass * (dT) / dt / 1e15

    ds_thetao = oet.load_glob(path_thetao)
    amoc_deep_dive.set_time_int(ds_thetao)
    lev_name = manual_msft.get_lev_coord(ds_thetao)
    if lev_name != "lev":
        print(ds_thetao[ds_thetao.variable_id].dims)
        ds_thetao = ds_thetao.rename(
            **{lev_name: "lev", f"{lev_name}_bnds": "lev_bnds"},
        )
        print(ds_thetao[ds_thetao.variable_id].dims)
    ds_thetao["lev_size"] = xr.DataArray(
        np.diff(ds_thetao["lev_bnds"])[:, 0],
        dims="lev",
    )
    a = ds_thetao.isel(time=0).load()
    tot_depth = a["cell_area"].copy()
    tot_depth[:] = 0
    for i, lev_d in enumerate(a["lev_size"].values):
        lev_slice = a["thetao"].isel(lev=i).load()
        tot_depth.data[~np.isnan(lev_slice.values)] += lev_d

    volume = a["cell_area"] * tot_depth
    res_thetao = amoc_deep_dive.BaseMaskedProducer(
        ds=ds_thetao,
        mask=mask_north_atlantic,
    ).get_mean_time_series()
    res_thetao_45 = amoc_deep_dive.BaseMaskedProducer(
        ds=ds_thetao,
        mask=mask_north_atlantic_n45,
    ).get_mean_time_series()

    time = ds_thetao["time"].values[1:]
    values_26 = calculate_heating_power_pw(
        np.diff(res_thetao[1]),
        mask=mask_north_atlantic,
        volume=volume,
    )
    values_45 = calculate_heating_power_pw(
        np.diff(res_thetao_45[1]),
        mask=mask_north_atlantic_n45,
        volume=volume,
    )
    plt.plot(time, smoother(values_26), label=rf"Ocean heating $>26^\circ$N{add_label}")
    plt.plot(time, smoother(values_45), label=rf"Ocean heating $>45^\circ$N{add_label}")
    return time, values_26, values_45


def pad_to_length(a, len_goal):
    if len_goal == len(a):
        return a
    return np.concatenate([a, [np.nan] * (len_goal - len(a))])


def make_plot(label, result_bucket, folder_dict, save_kw):
    # xr.Dataset(dict(bla=('time',np.ones(10))),
    #        coords=dict(time=np.arange(10)),
    #        )
    print("Load result")
    res = result_bucket[label]

    print("Load datasets")
    ds_hfds = oet.read_ds(
        folder_dict[label]["folder_hfds"],
        max_time=None,
        add_history=True,
    )

    ds_jenny = oet.load_glob(
        "/data/volume_2/tipping_figures/2024_05_07_amoc/data/transports_from_jenny/Atlantic_OHT_H_None_26.5N_SybrenPaper.nc",
    )

    s, ssp, m = label.split()
    da = ds_jenny.sel(members=f"6_{s}_{m}")
    v = np.concatenate([da["H_hist"].values, da[f"H_{ssp[3:]}"].values])
    v = v[~np.isnan(v)]
    t = np.arange(1850, 2501)

    res = result_bucket[label]
    mask_sum = res.masks[0]
    for m in res.masks[1:]:
        mask_sum = mask_sum | m
    mask_north_atlantic = atlantic_reference_mask(ds_hfds)

    mask_north_atlantic_n45 = mask_north_atlantic.copy()
    mask_north_atlantic_n45.data[mask_north_atlantic_n45.lat < 45] = 0
    ds_dict = dict(
        coords=dict(time=t),
        heat_transport_at_26=("time", pad_to_length(v, len(t))),
    )
    _c = ds_dict.pop("coords")
    _ = xr.Dataset(ds_dict, coords=_c)
    use_len = min(len(t), len(v))
    print("Start plotting")
    l = plt.plot(
        t[:use_len],
        smoother(v[:use_len]),
        label=r"AMOC heat transport at 26$^\circ$N",
    )
    # plt.scatter(
    #     t[:use_len],
    #     v[:use_len],
    #     c=l[0]._color,
    #     marker='.'
    # )
    value_dict = dict()
    for m, l in zip(
        res.masks
        + [
            mask_north_atlantic,
            mask_north_atlantic_n45,
            mask_sum,
        ],
        res.labels
        + [
            r"Heatloss North Atlantic and Arctic $>26^\circ$N",
            r"Heatloss North Atlantic and Arctic $>45^\circ$N",
            "Heatloss convective regions",
        ],
    ):
        if "Heatloss" not in l and "+" not in l:
            continue
        ds_sel = ds_hfds.where(m, drop=True)
        amoc_deep_dive.set_time_int(ds_sel)
        da = ds_sel["hfds"].astype(np.float64) * ds_hfds["cell_area"].astype(np.float64)
        plot_label = res.region_names.get(l, l)
        c = None

        legend_label = f'{plot_label} ({float(ds_hfds["cell_area"].where(m).sum()/1e12):.1f}$\\times10^{{6}}\\,\\mathrm{{km}}^2$)'
        aa = (da.sum(["lat", "lon"]) / -1e15).values
        value_dict[l] = aa
        ll = plt.plot(da["time"], smoother(aa), label=legend_label, c=c)
        # if '26' in l:
        #     plt.scatter(da["time"], aa, color=ll[0]._color, marker='.')

        ds_dict.update({l: ("time", pad_to_length(aa, len(t)))})

    _ = xr.Dataset(ds_dict, coords=_c)
    # print("Start plot_pw_old")
    # time, y26, y45 = plot_pw_old(
    #     folder_dict[label]["thetao_path"],
    #     mask_north_atlantic,
    #     mask_north_atlantic_n45,
    #     add_label=" old",
    # )
    # plt.plot(
    #     time,
    #     smoother(
    #         y26
    #         + value_dict["Heatloss North Atlantic and Arctic $>26^\circ$N"][
    #             -len(time) :
    #         ]
    #     ),
    #     label="Heatloss + ocean heating $>26^\circ$N (old)",
    #     ls="--",
    # )
    print("Start plot_pw_new")
    time, y26, y45 = plot_pw_new(
        folder_dict[label]["thetao_path"],
        mask_north_atlantic,
        mask_north_atlantic_n45,
        add_label="",
    )
    ds_dict.update(
        {
            "Ocean heating 26N": ("time", pad_to_length(y26, len(t))),
            "Ocean heating 45N": ("time", pad_to_length(y45, len(t))),
        },
    )
    y_plus = (
        y26
        + value_dict[r"Heatloss North Atlantic and Arctic $>26^\circ$N"][-len(time) :]
    )
    plt.plot(
        time,
        smoother(y_plus),
        label=r"Heatloss + ocean heating $>26^\circ$N",
        ls="--",
    )
    ds_dict.update(
        {
            "Heatloss + Ocean heating 26N": ("time", pad_to_length(y_plus, len(t))),
            "Ocean heating 45N": ("time", pad_to_length(y45, len(t))),
        },
    )

    plt.suptitle(" ".join(label.split(" ")[:2]), y=1.3)

    plt.ylabel("Total heat loss [PW]")

    plt.legend(
        **oet.utils.legend_kw(
            ncol=1,
        ),
    )
    # bbox_to_anchor=(1.02, 0, 0.7, 1),))
    plt.xlim(1850 - 5, time[-1] + 5)
    plt.ylim(-0.1, 1.25)
    print("Save fig")
    oet.utils.save_fig(f"heat_{label}", **save_kw, sub_dir="figure_s6")
    try:
        ds = xr.Dataset(ds_dict, coords=_c)
    except:
        for k, v in ds_dict:
            print(k, v)
        raise
    save_file = os.path.join(save_kw["save_in"], "files_s6", f"heat_{label}.nc")
    if os.path.exists(save_file):
        os.remove(save_file)
    os.makedirs(os.path.split(save_file)[0], exist_ok=True)
    ds.to_netcdf(save_file)


def main(label):
    base = "/data/volume_2/tipping_figures/2024_05_07_amoc/"
    config = amoc_deep_dive_group.read_config(f"{base}/config.json")
    oet.config.config.read_dict(config["config_update"])

    figures_folder = os.path.join(base, "paper", "v2025.03.17_test")
    save_kw = dict(save_in=figures_folder, dpi=300, file_types=("png",))

    oet.utils.setup_plt()
    _cc = oet.utils.get_plt_colors()
    del _cc[1:4]
    global_color_dict = {
        v: k
        for k, v in zip(
            _cc,
            [
                "Nordic Seas",
                "Irminger Sea",
                "Labrador Sea",
            ],
        )
    }
    global_color_dict
    folder_dict: ty.Dict[str, str] = amoc_deep_dive_group.read_config(
        config["read_from"],
    )
    for l in sorted(folder_dict):
        if folder_dict[l].get("folder_siconca") and not folder_dict[l].get(
            "folder_siconc",
        ):
            folder_dict[l]["folder_siconc"] = folder_dict[l].get("folder_siconca")
    make_plot(label, get_res_bucket(label, folder_dict), folder_dict, save_kw)


if __name__ == "__main__":
    label = sys.argv[1]
    print(label)
    try:
        main(label)
    except Exception as e:
        print(e)
        raise e
        sys.exit(1)
    print("done")
