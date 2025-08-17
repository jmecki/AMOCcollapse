#
import argparse
import glob
import logging
import os
import time
import typing as ty
from dataclasses import dataclass
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import optim_esm_tools as oet
import scipy.ndimage
import xarray as xr
from scipy.signal.windows import gaussian

from amoc_collapse_scripts import amoc_deep_dive
from amoc_collapse_scripts import amoc_deep_dive_group
from amoc_collapse_scripts import helper_scripts
from amoc_collapse_scripts import path_setup
from amoc_collapse_scripts.amoc_deep_dive import mask_to_full_mask

rename_dict = {
    "Greenland Sea": "Nordic Seas",
    "Norwegian Sea": "Nordic Seas",
    "North Atlantic Ocean": "Irminger Sea",
    "Baffin Bay": "Labrador Sea",
    "Davis Strait": "Labrador Sea",
}


def gausian_filter(y, n=100, mu=7.5):
    b = gaussian(n, mu)
    gauss = scipy.ndimage.convolve1d(y, b / b.sum())
    return gauss


def smooth_da(da, **kw):
    da = da.copy()
    da.data = gausian_filter(da.values, **kw)
    return da


def sel_atlantic(
    da,
    lat_max=60,
    lat_min=45,
    lon_west_min=70,
    lon_east_max=30,
    drop=True,
):
    da = da.copy()
    da = da.where(da.lat > lat_min, drop=drop)
    da = da.where(da.lat < lat_max, drop=drop)

    da = da.where((da.lon > (360 - lon_west_min)) | (da.lon < lon_east_max), drop=drop)
    return da


def set_lim(extent=0, lat_max=60, lat_min=45, lon_west_min=70, lon_east_max=30):
    plt.xlim(-lon_west_min - extent, lon_east_max + extent)
    plt.ylim(lat_min - extent, lat_max + extent)


def load_datasets(path):
    ds_bsurf = oet.load_glob(path)
    datasets = {
        k: oet.load_glob(v)
        for k, v in oet.utils.tqdm(list(ds_bsurf.attrs.items()))
        if os.path.exists(v)
    }
    return ds_bsurf, datasets


def cft_time_to_frac_year(x):
    return x.year + x.month / 12 + x.day / 365


cft_time_to_frac_year_np = np.vectorize(cft_time_to_frac_year)


def plot_monthly(ds_bsurf):
    b_s_surf_da = (ds_bsurf.b_s_surf).load()
    b_t_surf_da = (ds_bsurf.b_t_surf).load()
    time = cft_time_to_frac_year_np(ds_bsurf["time"].values)
    _mean = partial(
        oet.analyze.tools._weighted_mean_3d_numba,
        weights=ds_bsurf["cell_area"].values,
    )
    plt.plot(time, _mean(b_t_surf_da.values), label="$B^t_{surf}$")
    plt.plot(time, _mean(b_s_surf_da.values), label="$B^s_{surf}$")
    plt.plot(time, _mean((b_t_surf_da + b_s_surf_da).values), label="$B_{surf}$")
    # plt.axhline(0, ls='..', lw=1)
    plt.legend()
    plt.ylabel("buoyancy [J/kg]")


def plot_smoothed(ds_bsurf, smooth_kw=None):
    smooth_kw = smooth_kw or dict(mu=7.5 * 12, n=1000)
    b_s_surf_da = ds_bsurf.b_s_surf.load()
    b_t_surf_da = ds_bsurf.b_t_surf.load()
    time = cft_time_to_frac_year_np(ds_bsurf["time"].values)
    _mean = partial(
        oet.analyze.tools._weighted_mean_3d_numba,
        weights=ds_bsurf["cell_area"].values,
    )
    _mean_smooth = lambda x: gausian_filter(_mean(x), **smooth_kw)
    plt.plot(time, _mean_smooth(b_t_surf_da.values), label="$B^t_{surf}$")
    plt.plot(time, _mean_smooth(b_s_surf_da.values), label="$B^s_{surf}$")
    plt.plot(time, _mean_smooth((b_t_surf_da + b_s_surf_da).values), label="$B_{surf}$")
    # plt.axhline(0, ls='..', lw=1)
    plt.legend()
    plt.ylabel("buoyancy [J/kg]")


def sel_mask(a, m):
    return a.where(m, drop=True)


def plot_smoothed_regions(ds_bsurf, save_in, label, smooth_kw=None):
    smooth_kw = smooth_kw or dict(mu=7.5 * 12, n=1000)
    b_s_surf_da = ds_bsurf.b_s_surf.load()
    b_t_surf_da = ds_bsurf.b_t_surf.load()
    result_bucket = load_result_bucket(label)

    for region_label, mask in result_bucket.region_dict.items():
        mask_full = amoc_deep_dive.mask_to_full_mask(mask)

        time = cft_time_to_frac_year_np(ds_bsurf["time"].values)

        smoothed_weighted_mean = lambda x: gausian_filter(
            oet.analyze.tools._weighted_mean_3d_numba(
                x,
                weights=sel_mask(ds_bsurf["cell_area"], mask_full).values,
            ),
            **smooth_kw,
        )

        plt.plot(
            time,
            smoothed_weighted_mean(sel_mask(b_t_surf_da, mask_full).values),
            label="$B^t_{surf}$",
        )
        plt.plot(
            time,
            smoothed_weighted_mean(sel_mask(b_s_surf_da, mask_full).values),
            label="$B^s_{surf}$",
        )

        plt.plot(
            time,
            smoothed_weighted_mean(
                (sel_mask(b_t_surf_da + b_s_surf_da, mask_full)).values,
            ),
            label="$B_{surf}$",
        )
        # plt.axhline(0, ls='..', lw=1)
        plt.legend()
        plt.ylabel("buoyancy [J/kg]")
        plt.title(f"{result_bucket.region_names.get(region_label)} {label}")
        oet.plot_utils.save_fig(
            f"b_surf_reg_{region_label}_{result_bucket.region_names.get(region_label)}",
            save_in=save_in,
            dpi=200,
        )
        plt.clf()


def main(path: str, save_in: str, log: logging.Logger):
    log.warning(
        oet.utils.print_versions(
            ["optim_esm_tools", "amoc_collapse_scripts"],
            print_output=False,
            return_string=True,
        ),
    )
    oet.plotting.plot_utils.setup_plt()
    log.warning("Load data")
    ds_bsurf, datasets = load_datasets(path)
    log.warning("Plot monthly")
    plot_monthly(sel_atlantic(ds_bsurf))
    plt.title(
        f'{datasets["wfo"].source_id} {datasets["wfo"].experiment_id} {datasets["wfo"].variant_label} 45:60N 70W:30E',
    )
    oet.plot_utils.save_fig("b_surf_mothly", save_in=save_in, dpi=200)
    plt.clf()
    log.warning("Plot yearly")
    plot_smoothed(sel_atlantic(ds_bsurf))
    plt.title(
        f'{datasets["wfo"].source_id} {datasets["wfo"].experiment_id} {datasets["wfo"].variant_label} 45:60N 70W:30E',
    )
    oet.plot_utils.save_fig("b_surf_yearly", save_in=save_in, dpi=200)
    plt.clf()
    log.warning("Plot regions")
    label = " ".join(os.path.split(path)[-1].split(".")[0].split("_"))
    plot_smoothed_regions(ds_bsurf, save_in, label, smooth_kw=None)
    log.warning("Done")


def load_result_bucket(label):
    config = amoc_deep_dive_group.read_config(
        f"../../AMOCcollapse/notebooks/../data/mixed_layer_config.json",
    )
    oet.config.config.read_dict(config["config_update"])
    folder_dict: ty.Dict[str, str] = amoc_deep_dive_group.read_config(
        config["read_from"],
    )
    for l in sorted(folder_dict):
        if folder_dict[l].get("folder_siconca") and not folder_dict[l].get(
            "folder_siconc",
        ):
            folder_dict[l]["folder_siconc"] = folder_dict[l].get("folder_siconca")

    print(f"------------{label}--------------")
    ds_mlotst = oet.read_ds(
        folder_dict[label]["folder_mlotst"],
        add_history=True,
        max_time=None,
    ).load()
    amoc_deep_dive.set_time_int(ds_mlotst)

    result_bucket = amoc_deep_dive.plot_mlotst_cells_full_ret(
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build b-surf")
    parser.add_argument("--intermediate_file", default="month_merged.nc", type=str)

    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-s", "--save_in", type=str)
    parser.add_argument("--profile_memory", action="store_true")

    logger = oet.get_logger(__name__)

    # logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    args = parser.parse_args()

    main_kw = dict(
        log=logger,
        path=args.path,
        save_in=args.save_in,
    )
    if args.profile_memory:
        logger.info("profile memory")
        import memory_profiler
        import time

        t0 = time.time()
        try:
            mem_usage = memory_profiler.memory_usage(
                (
                    main,
                    (),
                    main_kw,
                ),
                max_iterations=1,
            )
        except Exception as e:
            logger.critical(
                f"raised critical error {e} for {main_kw}",
                exc_info=e,
            )
            mem_usage = [-1]
        t1 = time.time()
        _tm = f"{(t1-t0)/60:.2f} m ({(t1-t0)/3600:.1f} h)"
        logger.warning(
            f"Took {_tm:24} | Memory: max {max(mem_usage):.0f} MB, avg {sum(mem_usage)/len(mem_usage):.0f} MB\t",
        )
    else:
        try:
            main(**main_kw)
        except Exception as e:
            logger.critical(
                f"raised critical error {e} for {main_kw}",
                exc_info=e,
            )
    logger.warning("Done bye bye")
