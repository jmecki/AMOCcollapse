import glob
import itertools
import os
import time
import typing as ty

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numba
import numpy as np
import optim_esm_tools as oet
import pandas as pd
import scipy
import xarray as xr
from . import helper_scripts
from .amoc_deep_dive import set_time_int
from matplotlib.legend_handler import HandlerTuple


def year_mon_mean(ds):
    tb = "time_bnds" if "time_bnds" in ds else "time_bounds"
    dx = (ds[tb][:, 1] - ds[tb][:, 0]).values
    dx_sec = np.array([d.total_seconds() for d in dx]).reshape(len(dx) // 12, 12)
    d_tot = np.sum(dx_sec, axis=1)

    v = ds[ds.variable_id]
    va = v.values

    va_r = va.reshape(va.shape[0] // 12, 12, *va.shape[1:])
    #     time_r = dx_sec.reshape(len(dx_sec)//12, 12)

    va_avg = np.sum((va_r.T * dx_sec.T / d_tot).T, axis=1)

    a = ds
    a = a.groupby("time.year").mean()
    a[ds.variable_id].data = va_avg
    a = a.rename(dict(year="time"))
    return a


def build_ds(files, target_file, min_time=None, max_time=None, ma_window=10):
    if os.path.exists(target_file):
        return target_file
    from optim_esm_tools.analyze.pre_process import (
        sanity_check,
        _remove_duplicate_time_stamps,
        _check_time_range,
    )

    max_time = max_time or (9999, 12, 30)  # unreasonably far away
    min_time = min_time or (0, 1, 1)  # unreasonably long ago

    ds = xr.concat([oet.load_glob(f) for f in files], dim="time")
    ds = year_mon_mean(ds)
    ds = ds.sel(time=slice(min_time[0], max_time[0]))
    #     ds.attrs.update(dict(path=os.path.split(target_file)[0]))
    ds.to_netcdf(target_file)

    #     _check_time_range(target_file, max_time, min_time, ma_window)
    #     _remove_duplicate_time_stamps(target_file)
    sanity_check(ds)

    return target_file


def find_parent(p, ds=None):
    ds = ds or oet.load_glob(p)
    pn = os.path.split(p)[0]
    for update in (
        [ds.attrs["variant_label"], ds.attrs["parent_variant_label"]],
        [ds.attrs["activity_id"], ds.attrs["parent_activity_id"]],
        [ds.attrs["experiment_id"], ds.attrs["parent_experiment_id"]],
        [os.path.split(pn)[1], "*"],
    ):
        pn = pn.replace(*update)
    #     print(pn)
    return sorted(glob.glob(pn))


def find_paths(target, clean=False, need_pi=True):
    if (
        target
        == "/data/volume_2/2024_06_17_msft_manual_stage/data/CMIP6/ScenarioMIP/CCCma/CanESM5/ssp585/r1i1p1f1/Omon/msftmz/gn/v20240820/merged.nc"
    ):
        return dict(p_sc=target)
    h_h = find_parent(target)
    if not h_h or not len(h_h) == 1:
        raise ValueError(h_h)
    h_h = h_h[0]
    p_h = os.path.join(h_h, "merged.nc")
    if os.path.exists(p_h) and clean:
        os.remove(p_h)
    elif os.path.exists(p_h):
        # good, we want this path and don't clean
        pass
    else:
        # build the dataset
        f_h = sorted(glob.glob(os.path.join(h_h, "*.nc")))
        build_ds(f_h, p_h)
    if not need_pi:
        return dict(p_sc=target, p_h=p_h)
    h_pi = find_parent(p_h)
    if not h_pi or not len(h_pi) == 1:
        raise ValueError(h_pi)
    h_pi = h_pi[0]
    p_pi = os.path.join(h_pi, "merged.nc")
    if os.path.exists(p_pi) and clean:
        os.remove(p_pi)
    elif os.path.exists(p_pi):
        # good, we want this path and don't clean
        pass
    else:
        # build the dataset
        f_pi = sorted(glob.glob(os.path.join(h_pi, "*.nc")))
        build_ds(f_pi, p_pi)
    return dict(p_sc=target, p_h=p_h, p_pi=p_pi)


@numba.njit
def max_lat_lev_c(v):
    res = np.zeros(len(v), dtype=np.float64)
    for i, xy in enumerate(v):
        m = 0
        for x in xy:
            for y in x:
                m = max(m, y)
        res[i] = m
    return res


def get_basin_var(ds):
    for b_var in "basin 3basin".split():
        if b_var in ds or b_var in ds.dims:
            break
    else:
        raise ValueError(f"No basin variable in {ds.dims}")
    return b_var


def add_read_max(ds, variable_id=None):
    variable_id = variable_id or ds.variable_id
    v = ds[variable_id].values
    ds = ds.copy()
    b_var = get_basin_var(ds)
    for s_i, _ in enumerate(ds[b_var]):
        v_i = v[:, s_i].squeeze()
        ts = max_lat_lev_c(v_i)
        rm10 = helper_scripts.running_mean(ts, 10)

        ds[f"{variable_id}_basin_{s_i}"] = ("time", ts)
        ds[f"{variable_id}_basin_{s_i}"].attrs = ds[variable_id].attrs
        ds[f"{variable_id}_basin_{s_i}_run_mean_10"] = ("time", rm10)
        ds[f"{variable_id}_basin_{s_i}_run_mean_10"].attrs = ds[variable_id].attrs
    return ds


def max_jump(ds, ds_pi, field, nyear=10):
    mj = np.nanmax(np.abs(ds[field].values[nyear:] - ds[field].values[:-nyear]))
    pi_std = np.nanstd(ds_pi[field])
    if "fix" in field:
        f_fix = field.replace("_run_mean_10", "") + "ed"

        a = ds[f_fix].values
        b = ds_pi[f_fix].values
        if not np.array_equal(a, b):
            oet.get_logger().warning(f"Got {a} and {b} for indexes, this differs!")
    return mj / pi_std


def me_pi_ratio(ds, field, ds_pi):
    # oet.get_logger().warning("se_pi_ratio_single not 1:1 similar to 2d version")
    me = ds[field].max() - ds[field].values[-5]
    pi_std = np.nanstd(ds_pi[field])
    return float(abs(me / pi_std))


def mm_pi_ratio(ds, field, ds_pi):
    # oet.get_logger().warning("se_pi_ratio_single not 1:1 similar to 2d version")
    se = ds[field].max() - ds[field].min()
    pi_std = np.nanstd(ds_pi[field])
    return float(abs(se / pi_std))


def se_pi_ratio(ds, field, ds_pi, ma_window=10):
    # oet.get_logger().warning("se_pi_ratio_single not 1:1 similar to 2d version")
    se = ds[field].values[-ma_window] - ds[field].values[ma_window]
    pi_std = np.nanstd(ds_pi[field])
    return abs(se / pi_std)


def calculate_rho_mm(ds, field, rm_0, rm_1=None):
    ar = ds[field].values
    if len(ar.shape) > 1:
        ar = ds[field].mean("lat lon".split()).values
    ar_0 = helper_scripts.running_mean(ar, rm_0)
    ar_1 = ar if rm_1 is None else helper_scripts.running_mean(ar, rm_1)
    mm = ds[f"{field}_run_mean_10"].max() - ds[f"{field}_run_mean_10"].min()
    return abs(float(mm / np.nanstd(ar_1 - ar_0)))


def calculate_rho_me(ds, field, rm_0, rm_1=None):
    ar = ds[field].values
    if len(ar.shape) > 1:
        ar = ds[field].mean("lat lon".split()).values
    ar_0 = helper_scripts.running_mean(ar, rm_0)
    ar_1 = ar if rm_1 is None else helper_scripts.running_mean(ar, rm_1)
    me = ds[f"{field}_run_mean_10"].max() - ds[f"{field}_run_mean_10"].values[-5]
    return abs(float(me / np.nanstd(ar_1 - ar_0)))


def calculate_rho(ds, field, rm_0, rm_1=None, ma_window=10):
    ar = ds[field].values
    if len(ar.shape) > 1:
        ar = ds[field].mean("lat lon".split()).values
    ar_0 = helper_scripts.running_mean(ar, rm_0)
    ar_1 = ar if rm_1 is None else helper_scripts.running_mean(ar, rm_1)
    se = ds[field].values[-ma_window] - ds[field].values[ma_window]
    return abs(float(se / np.nanstd(ar_1 - ar_0)))


def shift_to_start(ds, ds_ref, offset=0):
    if ds is None:
        return ds
    t0 = ds_ref["time"].values[0]
    if isinstance(t0, np.integer):
        new_time = np.arange(t0 - len(ds["time"]), t0) - offset
    #         print(new_time)
    else:
        new_time = [
            oet.analyze.xarray_tools._native_date_fmt(
                ds_ref["time"].values,
                (t0.year - offset - dy, 7, 1),
            )
            for dy in list(range(len(ds["time"])))[::-1]
        ]
    ds = ds.copy()
    ds["time"] = new_time
    return ds


def plot_rev(
    ds2,
    ds2_pi_shift,
    min_max="max",
    add_pi=None,
    start_at_zero=False,
    fields=None,
):
    # plt.show()
    _var = ds2.variable_id
    cs = {}
    add_pi = add_pi or (False if ds2_pi_shift is None else True)
    sectors = sorted(sectors_normalize(ds2))

    if fields is None:
        min_max = min_max or "min max".split()
        min_max = oet.utils.to_str_tuple(min_max)
        fields = [
            f"{sec}_{min_max}_vari" if sec != "atlantic" else f"{sec}_{min_max}_fix"
            for sec, min_max in itertools.product(sectors, min_max)
        ]
    for _vv in fields:
        _vvrm = _vv + "_run_mean_10"

        offset = ds2[_vvrm].values[5] if start_at_zero else 0

        _ = (ds2[_vv] - offset).plot(alpha=0.5)
        (ds2[_vvrm] - offset).plot(label=_vv, c=_[0].get_color())

        if add_pi:
            (ds2_pi_shift[_vv] - offset).plot(c=_[0].get_color(), alpha=0.5)
            (ds2_pi_shift[_vvrm] - offset).plot(c=_[0].get_color())

    plt.legend(ncol=2, loc="upper right")


def plot_rev_per_sector(ds2, ds2_pi_shift):
    # plt.show()
    _var = ds2.variable_id
    cs = {}
    sectors = get_sectors(ds2_pi_shift)
    for i, sec in enumerate(sectors):
        _vv = f"{_var}_basin_{i}"
        _vvrm = f"{_var}_basin_{i}_run_mean_10"

        if ds2_pi_shift is not None:
            cs[_vv] = ds2_pi_shift[_vv].plot(label=f"{sec} ({i})")
            cs[_vvrm] = ds2_pi_shift[_vvrm].plot()

    for i, sec in enumerate(sectors):
        _vv = f"{_var}_basin_{i}"
        _vvrm = f"{_var}_basin_{i}_run_mean_10"

        ds2[_vv].plot(c=cs[_vv][0].get_color())
        ds2[_vvrm].plot(c=cs[_vvrm][0].get_color())

    plt.legend(ncol=2, loc="upper right")


# def tabulated_plot(ds2, ds2_pi_shift):
#     #     h='/data/volume_2/synda/stage/ScenarioMIP/MOHC/UKESM1-0-LL/ssp585/r4i1p1f2/Omon/msftyz/gn/v20210205'
#     #     files = sorted([f for f in glob.glob(os.path.join(h, '*.nc')) if 'merged.nc' not in f])
#     #     target = os.path.join(h, 'merged.nc')
#     #     build_ds(files, target)
#     #     paths = find_paths(target)
#     _, axes = plt.subplots(2, 1, dpi=300, gridspec_kw=dict(height_ratios=[4, 1]))
#     plt.sca(axes[0])
#     #     ds2, ds2_pi = get_dss_cached(paths)
#     #     ds2_pi_shift = shift_to_start(ds2_pi, ds2, offset=50)
#     plot_rev_per_sector(ds2, ds2_pi_shift)

#     _df_docs = []
#     sectors = get_sectors(ds2)
#     if len(np.array(sectors).shape) > 1:
#         sectors = sectors[0]
#     doc = {
#         f"basin_{i}": is1d_summarize_stats(
#             ds2,
#             ds2_pi_shift,
#             field=f"{ds2.variable_id}_basin_{i}",
#         )
#         for i, _ in enumerate(sectors)
#     }

#     for b, v in doc.items():
#         _df_docs.append(dict(**v, basin=b))
#     _df_docs = pd.DataFrame(_df_docs)
#     ax = axes[1]
#     # ax = ax or plt.gcf().add_subplot(2, 1, 2)
#     ax.axis("off")
#     ax.axis("tight")
#     _df_docs = _df_docs.set_index("basin")
#     use_formats = helper_scripts.table_formats.copy()
#     for (
#         k
#     ) in "e_s se_gl_r se_gl_rold j_gl_se se_trop_r se_pi_ratio rho10_50 se j_gl_std se_gl_std se_std_trop".split():
#         del use_formats[k]

#     for k, f in use_formats.items():
#         if k in _df_docs:
#             _df_docs[k] = _df_docs[k].map(f"{{:,{f}}}".format)

#     for k in _df_docs.columns:
#         if k not in use_formats:
#             del _df_docs[k]
#     table = ax.table(
#         #     cellText=_df_docs.values,
#         cellText=[
#             [helper_scripts.scientific_latex_notation(f.replace("%", r"\%")) for f in ff]
#             for ff in _df_docs.values
#         ],
#         rowLabels=_df_docs.index,
#         colLabels=_df_docs.columns,
#         #     cellColours=[
#         #         [(pass_color if v else [1, 1, 1]) for v in row] for row in tips.values
#         #     ],
#         colWidths=[0.15] * len(_df_docs.columns),
#         loc="bottom",
#         colLoc="center",
#         rowLoc="center",
#         cellLoc="center",
#     )
#     table.set_fontsize(16)
#     return doc


def tabulated_plot(ds2, ds2_pi_shift, add_delta=True, fields=None):
    if add_delta:
        _, axes_dict = plt.subplot_mosaic(
            "AAABBCCC\nDDDDDDDD",
            gridspec_kw=dict(height_ratios=[1, 1 / 6]),
            dpi=300,
            figsize=(18, 6),
        )
        axes = axes_dict["A"], axes_dict["D"]
    else:
        _, axes = plt.subplots(2, 1, dpi=300, gridspec_kw=dict(height_ratios=[4, 1]))
    sectors = sectors_normalize(ds2)
    if isinstance(fields, ty.Iterable) and not isinstance(fields, ty.Mapping):
        fields = {k: k for k in fields}
    if fields is None:
        fields = {
            f"{sec}_{min_max}": (
                f"{sec}_{min_max}_vari" if sec != "atlantic" else f"{sec}_{min_max}_fix"
            )
            for sec, min_max in itertools.product(sectors, "min max".split())
        }

    plt.sca(axes[0])
    plot_rev(ds2, ds2_pi_shift, fields=fields.values())
    plt.ylabel(rf"$\mathrm{{{ds2.variable_id}}}\ [\mathrm{{Sv}}]$")
    plt.title("")

    if add_delta:
        axes_dict["A"].get_legend().remove()
        plt.sca(axes_dict["C"])
        plot_rev(
            ds2,
            ds2_pi_shift,
            add_pi=False,
            start_at_zero=True,
            fields=fields.values(),
        )
        plt.ylabel(rf"$\Delta \mathrm{{{ds2.variable_id}}}\ [\mathrm{{Sv}}]$")
        plt.title("")
        lines = list(axes_dict["A"].get_lines())
        labels = [a.get_label() for a in lines]
        keep = [not l.startswith("_") for l in labels]
        lines = [l for l, k in zip(lines, keep) if k]
        labels = [l for l, k in zip(labels, keep) if k]
        plt.sca(axes_dict["B"])
        plt.legend(
            lines,
            labels,
            **oet.utils.legend_kw(
                loc="upper left",
                ncol=1,
                mode=None,
                bbox_to_anchor=(0.0, 0, 1, 1),
            ),
        )
        plt.axis("off")
        axes_dict["C"].get_legend().remove()
    else:
        plt.legend(**oet.utils.legend_kw(ncol=2))

    _df_docs = []

    doc = {
        k: is1d_summarize_stats(
            ds2,
            ds2_pi_shift,
            field=v,
        )
        for k, v in fields.items()
    }
    for sec, v in doc.items():
        _df_docs.append(dict(**v, sec=sec))
    _df_docs = pd.DataFrame(_df_docs)
    ax = axes[1]
    # ax = ax or plt.gcf().add_subplot(2, 1, 2)
    ax.axis("off")
    ax.axis("tight")
    _df_docs = _df_docs.set_index("sec")
    use_formats = helper_scripts.table_formats.copy()
    use_formats.update({"me_pi_std": ".2g"})
    for (
        k
    ) in "e_s se_gl_r se_gl_rold j_gl_se se_trop_r se_pi_ratio rho10_50 se j_gl_std se_gl_std se_std_trop".split():
        if k in use_formats:
            del use_formats[k]
    for k, f in use_formats.items():
        if k in _df_docs:
            _df_docs[k] = _df_docs[k].map(f"{{:,{f}}}".format)

    for k in _df_docs.columns:
        if k not in use_formats:
            del _df_docs[k]
    #     print(doc)
    table = ax.table(
        #     cellText=_df_docs.values,
        cellText=[
            [helper_scripts.scientific_latex_notation(f.replace("%", r"\%")) for f in ff]
            for ff in _df_docs.values
        ],
        rowLabels=_df_docs.index,
        colLabels=_df_docs.columns,
        cellColours=[
            [((0.75, 1, 0.75) if crit_E(**d) else [1, 1, 1])] * len(_df_docs.columns)
            for d in doc.values()
        ],
        colWidths=[0.13] * len(_df_docs.columns),
        loc="bottom",
        colLoc="center",
        rowLoc="center",
        cellLoc="center",
    )
    table.set_fontsize(16)
    return doc


def get_sectors(ds):
    sectors = ds.sector.values if "sector" in ds else None
    if sectors is None and "region" in ds:
        sectors = ds.region.values
    if sectors is not None and len(sectors.shape) > 1:
        sectors = sectors[0]
    guess = "Global Atlantic Indo-Pacific".split()
    if sectors is None and all(
        any(g in v for v in ds[ds.variable_id].attrs.values()) for g in guess
    ):
        sectors = guess
    elif sectors is None:
        b_var = get_basin_var(ds)
        if (
            ds[b_var].attrs.get("long_name")
            == "Sub-basin mask (1=Global 2=Atlantic 3=Indo-Pacific)"
        ):
            return "Global Atlantic Indo-Pacific".split()
        if (
            ds[b_var].attrs.get("requested")
            == "atlantic_arctic_ocean=0, indian_pacific_ocean=1, global_ocean=2"
        ):
            return "atlantic_arctic_ocean indian_pacific_ocean global_ocean".split()
        raise ValueError
    sectors = [sec.decode() if isinstance(sec, bytes) else sec for sec in sectors]
    return sectors


def sector_normalize(sector: ty.Union[bytes, str]) -> str:
    if isinstance(sector, bytes):
        sector = sector.decode()

    sector = sector.strip(" ").strip("\n")
    if sector in ["a", "atlantic_arctic_ocean", "Atlantic"]:
        return "atlantic"
    if sector in ["atlantic_arctic_extended_ocean"]:
        return "atl_ext"
    if sector in ["i", "indian_pacific_ocean", "Indo-Pacific"]:
        return "indian_pa"
    if sector in ["pacific_ocean"]:
        return "pacific"
    if sector in ["g", "global_ocean", "Global"]:
        return "global"
    raise ValueError(sector)


def sectors_normalize(ds):
    sectors = get_sectors(ds)
    return [sector_normalize(s) for s in sectors]


@numba.njit
def min_max_lat_lev_idx(v):
    res_min = np.zeros((len(v), 2), dtype=np.int64)
    res_max = np.zeros((len(v), 2), dtype=np.int64)
    # Infinite min/max for all practival purposes

    for i, xy in enumerate(v):
        m_max = -np.inf
        m_min = np.inf
        for x_i, x in enumerate(xy):
            for y_i, value in enumerate(x):
                if value > m_max:
                    m_max = value
                    res_max[i] = x_i, y_i
                if value < m_min:
                    m_min = value
                    res_min[i] = x_i, y_i
    return res_min, res_max


@numba.njit
def index_2d(a, idx_2d):
    res = np.zeros(len(a), dtype=a.dtype)
    for i in range(len(a)):
        res[i] = a[i][idx_2d[i][0], idx_2d[i][1]]
    return res


def fix_j_mean_format(ds):
    pattern = (
        f"/data/volume_2/synda/year_data_merged/CMIP6/*/"
        f"{ds.institution_id}/{ds.source_id}/*/{ds.variant_label}/*/sos/{ds.grid_label}/*/merged.nc"
    )
    matches = glob.glob(pattern)
    for f in matches:
        print(f"trying {f} from {len(matches)} matches")
        try:
            ds_sos = oet.load_glob(matches[0])
            assert "lat" in ds_sos, (ds_sos.dims, matches[0])

            ds_new = ds.copy()
            ds_new = ds_new.drop_vars("msftyz")
            ds_new = ds_new.drop_dims("j-mean")
            idx = np.median(np.argmin(np.abs(ds_sos["lat"].values - 26.5), axis=0))
            a = ds_sos["lat"].isel(x=int(idx)).values
            ds_new["lat"] = xr.DataArray(a, dims=["lat"])

            ds_new["msftyz"] = xr.DataArray(
                ds["msftyz"].values,
                dims=[(d if d != "j-mean" else "lat") for d in ds["msftyz"].dims],
            )
            ds_new = ds_new.drop_duplicates("lat")
            break
        except Exception as e:
            print(f"{f} did not work because of {e}")
    else:
        raise ValueError
    return ds_new


def normalize(ds):
    if "nav_lat" in ds and "y" in ds[ds.variable_id].dims:
        ds = ds.assign_coords(dict(y=("y", ds["nav_lat"].values.squeeze()))).rename(
            dict(y="rlat"),
        )
        ds = ds.drop_duplicates("rlat")
    if "j-mean" in ds:
        ds = fix_j_mean_format(ds)
    return ds


def get_rlat_coord(ds):
    for k in "rlat lat y".split():
        if k in ds.dims:
            return k
    for k in "rlat lat y".split():
        if k in ds:
            return k
    raise ValueError


def get_lev_coord(ds):
    for k in "lev olevel".split():
        if k in ds or k in ds.dims:
            return k
    raise ValueError


def add_field(ds, target_field, data, dims=("time"), attrs_from_field=None):
    ds[target_field] = (dims, data)
    if attrs_from_field:
        ds[target_field].attrs = ds[attrs_from_field].attrs.copy()
    return ds


def extract_fields(ds, variable_id=None, reference_time_slice=slice(100, 150)):
    variable_id = variable_id or ds.variable_id
    # v = ds[variable_id].values
    ds = ds.copy()
    b_var = get_basin_var(ds)
    sectors = [sector_normalize(s) for s in get_sectors(ds)]

    assert len(ds[b_var]) == len(sectors)
    ds[variable_id] = ds[variable_id] / 1e9
    ds[variable_id].attrs.update(dict(units="Sv"))
    for s_i, sector in enumerate(sectors):
        da_sel = ds[variable_id].isel(**{b_var: s_i})

        if "sector" in ds or "region" in ds:
            key = "sector" if "sector" in ds else "region"
            _sectors = ds[key].values[s_i]
            if isinstance(_sectors, np.ndarray):
                _sectors = ds[key].values[0, s_i]
            assert sector_normalize(_sectors) == sector, (ds[key].values[0], sector)

        lat_c = get_rlat_coord(ds)
        if sector == "atlantic":
            da_sel = da_sel.sel(**{lat_c: 26.5}, method="nearest")

        if sector == "indian_pa":
            da_sel = da_sel.sel(**{lat_c: slice(-30, 30)})

        if sector == "global":
            da_sel = da_sel.sel(**{lat_c: slice(None, -30)})

        v_i = da_sel.values.squeeze()

        if sector == "atlantic":
            assert len(v_i.shape) == 2
            len_time, len_lev = v_i.shape
            v_i = v_i.reshape(len_time, len_lev, 1)

        a = v_i
        min_2d, max_2d = min_max_lat_lev_idx(a)
        v_var_max = index_2d(a, max_2d)
        v_var_min = index_2d(a, min_2d)

        a_50_mean = a[reference_time_slice].mean(axis=0)
        if np.all(np.isnan(a_50_mean)):
            oet.get_logger().warning(
                f"No data for {sector} ({ds.source_id}) - everything is nan?",
            )
            assert np.all(np.isnan(v_var_max))
            min_1d = min_2d[0]
            max_1d = min_2d[0]

            v_fixed_max = v_var_max
            v_fixed_min = v_var_max
        else:
            min_1d = np.argwhere(a_50_mean == np.nanmin(a_50_mean))[0]
            max_1d = np.argwhere(a_50_mean == np.nanmax(a_50_mean))[0]

            v_fixed_max = a[:, max_1d[0], max_1d[1]]
            v_fixed_min = a[:, min_1d[0], min_1d[1]]

        _var = variable_id
        kw = dict(dims="time", attrs_from_field=_var)
        # Variable fields
        add_field(ds, f"{sector}_max_vari", v_var_max, **kw)
        add_field(
            ds,
            f"{sector}_max_vari_run_mean_10",
            helper_scripts.running_mean(v_var_max, 10),
            **kw,
        )
        add_field(ds, f"{sector}_min_vari", v_var_min, **kw)
        add_field(
            ds,
            f"{sector}_min_vari_run_mean_10",
            helper_scripts.running_mean(v_var_min, 10),
            **kw,
        )

        # Fixed fields
        add_field(ds, f"{sector}_max_fix", v_fixed_max, **kw)
        add_field(
            ds,
            f"{sector}_max_fix_run_mean_10",
            helper_scripts.running_mean(v_fixed_max, 10),
            **kw,
        )
        add_field(ds, f"{sector}_min_fix", v_fixed_min, **kw)
        add_field(
            ds,
            f"{sector}_min_fix_run_mean_10",
            helper_scripts.running_mean(v_fixed_min, 10),
            **kw,
        )

        # Coord fields
        add_field(ds, f"{sector}_min_2d", min_2d, dims=("time", "idx_2d"))
        add_field(ds, f"{sector}_max_2d", max_2d, dims=("time", "idx_2d"))
        add_field(ds, f"{sector}_min_fixed", min_1d, dims=("idx_2d"))
        add_field(ds, f"{sector}_max_fixed", max_1d, dims=("idx_2d"))
    return ds


def get_dss_cached(
    paths,
    head,
    base,
    name="pre_processed_v0.1.2.nc",
    sector_vars="region sector".split(),
):
    p_sc = paths["p_sc"]

    t_ds = p_sc.replace(head, base).replace("merged.nc", name)
    if os.path.exists(t_ds):
        ds2 = oet.load_glob(t_ds)
    else:
        p_h = paths["p_h"]
        ds = xr.concat([oet.load_glob(p_h), oet.load_glob(p_sc)], dim="time")
        ds = normalize(ds)
        ds2 = extract_fields(ds)
        os.makedirs(os.path.split(t_ds)[0], exist_ok=True)
        ds2.to_netcdf(t_ds)
    for v in sector_vars:
        if v in ds2.coords and "time" in ds2[v].coords:
            ds2[v] = ds2[v].isel(time=-1)

    p_pi = paths.get("p_pi", None)
    if p_pi is not None:
        t_pi = p_pi.replace(head, base).replace("merged.nc", name)
        if os.path.exists(t_pi):
            ds2_pi = oet.load_glob(t_pi)
        else:
            ds_pi = oet.load_glob(p_pi)
            for v in sector_vars:
                if v in ds2 and v not in ds_pi:
                    ds_pi[v] = ds2[v].copy()
                    #             ds_pi['sector']=ds_pi['sector'].drop_dims('time')
                    print(ds2[v])
            ds_pi = normalize(ds_pi)
            ds2_pi = extract_fields(ds_pi)
            os.makedirs(os.path.split(t_pi)[0], exist_ok=True)
            ds2_pi.to_netcdf(t_pi)
    else:
        ds2_pi = None

    return ds2, ds2_pi


def plot_rev_simple(ds2, fields=None, **kw):
    _var = ds2.variable_id
    for _vv in fields:
        _vvrm = _vv + "_run_mean_10"
        lab = kw.pop("label")
        ds2[_vvrm].plot(**kw, label=lab)
        ds2[_vv].plot(**kw, alpha=0.5)


def match_docs(doc1, doc2, keys=None):
    keys = keys or "source_id ssp variant_label variable_id basin".split()
    for k in keys:
        if doc1.get(k) != doc2.get(k):
            return False
    return True


def summary_of_variant_labels(p_interest, p_pass, all_docs):
    doc = [d for d in all_docs if d["h"] == p_interest][0]

    tot_docs = [
        d
        for d in all_docs
        if d["h"] in p_pass
        and match_docs(d, doc, keys="source_id ssp variable_id basin".split())
    ]
    matches = [
        d
        for d in all_docs
        if match_docs(d, doc, keys="source_id ssp variable_id basin".split())
    ]

    filtered_matches = []
    for m in matches:
        if any(match_docs(d, m) for d in filtered_matches):
            continue
        filtered_matches.append(m)

    found_in = [r["variant_label"] for r in tot_docs]
    last_year = [r["last_year"] for r in tot_docs]
    criteria = ["E" for r in tot_docs]

    assert len(np.unique(found_in)) == len(found_in), found_in

    total_available = len(filtered_matches)
    p2100_available = sum(d["last_year"] > 2100 for d in filtered_matches)

    return dict(
        found_in=found_in,
        total_available=total_available,
        p2100_available=p2100_available,
        found_in_year=last_year,
        criteria=criteria,
    )


def get_depth(ds2):
    depth = ds2[get_lev_coord(ds2)][int(round(ds2["atlantic_max_fixed"].values[0], 0))]
    if depth.attrs["units"] == "centimeters":
        return float(depth) / 1e2
    else:
        ...
    return float(depth)


order = [
    "criteria",
    "max_jump",
    "p_dip",
    "me_pi_std",
    "$\\text{depth [m]}$",
    "label",
    "version",
]


def rename_key(k):
    kn = {
        "p_sym_mi": "p_\\text{sim}",
        "p_dip": "p_\\text{dip}",
        "J2_rm": "J2",
        "J2_min0_rm": "J2_\\text{min}",
        #      'variable_id': 'var',
        "max_jump": "{MJ_\\text{avg}}/{ \\sigma_{\\text{PiC}} } ",
        #      'me_std_trop':  '\\dfrac{ME}{\\bar{\\sigma}_\\text{trop std}}',
        #      'se_std_trop': '\\dfrac{SE}{\\bar{\\sigma}_\\text{trop std}}' ,
        "me_std_trop": "{ME}/{\\bar{\\sigma}_\\text{trop std}}",
        "se_std_trop": "{SE}/{\\bar{\\sigma}_\\text{trop std}}",
        "me_pi_std": "{ME}/{ \\sigma_{\\text{PiC}} }",
        "e_s": "ES_{S.I.}",
        #      'rho10_50': '\dfrac{SE}{\\rho_{10}^{50}}',
        "rho10_50": "{SE}//{\\rho_{10}^{50}}",
        "j_gl_std": "MJ_\\text{avg} / \\bar{\\sigma}_\\text{glob std}",
        "area_sum": "\\text{area [km]}^2",
    }.get(k, None)
    if kn is None:
        return k
    return f"${kn}$"


def get_exp(h):
    return h.split(os.sep)[9]


def get_ver(h):
    return h.split(os.sep)[-1]


def get_vari(h):
    return h.split(os.sep)[10]


def write_summary_df(h, ds, ds_pi, field):
    calc = is1d_summarize_stats(ds, ds_pi, field=field)

    keep_kw = {
        rename_key(k): helper_scripts.scientific_latex_notation(
            f'{v: {use_formats.get(k,"s")}}',
        ).replace("%", r"\%")
        for k, v in calc.items()
        if k in order
    }
    keep_kw["$\\text{depth [m]}$"] = helper_scripts.scientific_latex_notation(
        f"{get_depth(ds):.2g}",
    )
    summary = {
        get_exp(h): dict(
            criteria="E",
            label=get_vari(h),
            version=get_ver(h),
            **keep_kw,
        ),
    }

    df = pd.DataFrame(summary)
    return df


def df_to_latex(df, **kw):
    bla = df.fillna("")
    bla = bla[[col for col in [rename_key(o) for o in order] if col in bla.columns]]
    return bla.sort_index().T.to_latex(**kw)


def add_figure_format(
    figure_name,
    fig_place="figures/streamfunctions",
    caption=None,
    size=0.5,
):
    caption = caption or figure_name
    fmt = f"""
\\begin{{figure}}[h!]
\\centering
 \\includegraphics[width={size}\\textwidth]{{{fig_place}/{figure_name}.png}}
 \\caption{{ {caption} }}
 \\label{{fig:{figure_name}}}
\\end{{figure}}
"""
    return fmt


def append(call_name, base, f_name, ds2, dfs, loc):
    plt.ylabel(rf"$\mathrm{{{ds2.variable_id}}}\ [\mathrm{{Sv}}]$")
    plt.title("")
    plt.legend(**oet.utils.legend_kw())

    oet.utils.save_fig(
        save_in=os.path.join(base, "figures_presentation_v10"),
        file_types=("png",),
        sub_dir="",
        name=f_name,
    )
    plt.show()

    conc = pd.concat(dfs)

    label = f_name
    doc_kw = dict(
        label=f"tab:{label}",
        caption=f"{call_name} {ds2.variable_id}",
        position="h!",
    )
    return f"""
    \\subsection{{{call_name}\\label{{sec:{label}}}}}
    \\subsubsection{{{ds2.variable_id} {loc} basin}}
    {call_name} is in {loc} and corresponds to \\autoref{{fig:{label}}} and \\autoref{{tab:{label}}}.
    {df_to_latex(conc, **doc_kw)}
    {add_figure_format(f_name, caption=f"{call_name} {ds2.variable_id}", size=0.7)}
    \\clearpage
    """


def to_share(ds2, ds2_pi, dest, field=None):
    field = field or "atlantic_max"
    da_pi = ds2_pi[f"{field}_fix"].copy().rename(time="time_pi")
    lc = get_rlat_coord(ds2)

    ds_share = xr.Dataset(
        {
            ds2.variable_id: ds2[f"{field}_fix"],
            ds2.variable_id + "_pi": da_pi,
            "depth": xr.DataArray(
                [get_depth(ds2)],
                dims=["depth"],
                attrs=dict(units="m"),
            ),
            lc: xr.DataArray(
                [26.5],
                dims=[lc],
                attrs=ds2[lc].attrs,
            ),
        },
        attrs=ds2.attrs,
    )
    ds_share.to_netcdf(dest)


def get_model(h):
    return h.split(os.sep)[8]


def slice_to_string(sl):
    return f"${sl.start} - {sl.stop}$"


def set_depth_meter(ds):
    coord = get_lev_coord(ds)
    depth = ds[coord]
    if depth.attrs["units"] == "centimeters":
        ds[coord] = ds[coord] / 100
        ds[coord].attrs["units"] = "m"


class EnsambleFromYear:
    def __init__(self, ds_group, lev_grid, lat_grid):
        self.ds_group = ds_group
        self.lev_grid = lev_grid
        self.lat_grid = lat_grid

    def get_grid(self):
        a, b = np.meshgrid(self.lev_grid, self.lat_grid)
        grd = np.array([(x, y) for x, y in zip(a.flatten(), b.flatten())])
        return grd

    def get_da_from_ds(self, ds2, year_slice):
        ds_n = ds2.copy()
        ds_s = ds_n.isel(
            **{
                get_basin_var(ds_n): np.argwhere(
                    np.array(sectors_normalize(ds_n)) == "atlantic",
                ).squeeze(),
            },
        )

        set_time_int(ds_n)

        ds_s[get_lev_coord(ds_n)] = -ds_s[get_lev_coord(ds_n)]
        #         depth = get_depth(ds_n)
        # TODO fix time_slice
        if year_slice == slice(2190, 2200) and ds_s["time"].values[-1] < 2181:
            oet.get_logger().warning(f"Manually shifting slice for {ds2.source_id}")
            da = ds_s[ds_s.variable_id].sel(time=slice(2170, 2180))
        else:
            da = ds_s[ds_s.variable_id].sel(time=year_slice)

        da = da.mean("time")

        return da

    def ensamble_from_year(self, **kw):
        res = self._ensamble_from_year(**kw)
        res_da = xr.DataArray(
            data=res,
            coords=dict(lev=self.lev_grid, lat=self.lat_grid),
        )
        return res_da

    def _ensamble_from_year(
        self,
        year_slice,
        _plot_intermediate=False,
        normalize_to_year=None,
        _reference_kw=None,
    ):
        _sum = None

        for ds2 in self.ds_group:
            this_itp = self._interpolate_ds(ds2, year_slice, _plot_intermediate)
            if normalize_to_year is not None:
                _reference_kw = _reference_kw or dict()
                norm = self._refernce_value(ds2, normalize_to_year, **_reference_kw)
                this_itp /= norm
            if _sum is None:
                _sum = this_itp / len(self.ds_group)
            else:
                _sum += this_itp / len(self.ds_group)
        return _sum

    def _refernce_value(self, ds2, year_slice, lat=26.5, depth=None):
        da = self.get_da_from_ds(ds2, year_slice=year_slice)
        depth = depth or get_depth(ds2)
        return float(
            da.sel(
                **{get_lev_coord(ds2): -depth, get_rlat_coord(ds2): lat},
                method="nearest",
            ),
        )

    def _interpolate_ds(self, ds2, year_slice, _plot_intermediate=False):
        grd = self.get_grid()
        da = self.get_da_from_ds(ds2, year_slice=year_slice)
        itp = scipy.interpolate.RegularGridInterpolator(
            [da[get_lev_coord(ds2)], da[get_rlat_coord(ds2)]],
            da.values,
            method="linear",
            bounds_error=False,
        )
        this_itp = itp(grd).reshape(len(self.lat_grid), len(self.lev_grid)).T
        if _plot_intermediate:
            plt.imshow(
                this_itp[::-1],
                extent=[
                    self.lat_grid[0],
                    self.lat_grid[-1],
                    self.lev_grid[0],
                    self.lev_grid[-1],
                ],
                aspect="auto",
            )
            plt.title(f"{ds2.source_id} {ds2.variable_id}")
            plt.colorbar(orientation="horizontal")
            plt.show()
        return this_itp


def arbitrary_cmap_handle(cmap, v_list, ax=None, **plt_kw):
    plt_kw = plt_kw or dict()
    ax = ax or plt.gca()
    lines = []
    for v in v_list:
        (l,) = ax.plot(np.zeros(2) * np.nan, np.zeros(2) * np.nan, c=cmap(v), **plt_kw)
        lines.append(l)
    return tuple(lines)


def cmap_legend(
    cmaps,
    labels,
    v_list,
    extra_handles: ty.Optional[list] = None,
    ax=None,
    plt_kw=None,
    **kw,
):
    kw = kw.copy()
    kw.setdefault("handlelength", 3)
    kw.setdefault("borderpad", 0.7)
    kw.setdefault("labelspacing", 0.4)
    ax = ax or plt.gca()
    handles = []
    for cm in cmaps:
        handles.append(arbitrary_cmap_handle(cm, v_list, **plt_kw))
    extra_handles = extra_handles or []
    handles += extra_handles
    ax.legend(
        handles=handles,
        labels=labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        **kw,
    )


def set_share_axes(axs, target=None, sharex=False, sharey=False):
    """Thanks https://stackoverflow.com/a/51684195/18280620"""
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_axes["x"].join(target, ax)
        if sharey:
            target._shared_axes["y"].join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1, :].flat:
            ax.xaxis.set_tick_params(which="both", labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
            plt.setp(ax.get_xticklabels(), visible=False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:, 1:].flat:
            ax.yaxis.set_tick_params(which="both", labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)


def annotate(
    label,
    x,
    y,
    x_text=None,
    y_text=None,
    use_text_color_for_edge=False,
    use_text_color_for_line=False,
    **kw,
):
    x_text = x_text or x
    y_text = y_text or y
    tkw = dict(
        size=10,
        horizontalalignment="center",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="gainsboro", alpha=1),
        xycoords="data",
        textcoords="data",
        arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.3"),
    )
    bbox = kw.pop("bbox", None)
    tkw.update(kw)
    if bbox is not None:
        tkw["bbox"].update(bbox)
    if use_text_color_for_edge:
        tkw["bbox"].update(edgecolor=kw.get("color", "k"))
    an = plt.gca().annotate(
        label,
        xy=(x, y),
        xytext=(x_text, y_text),
        **tkw,
    )
    if use_text_color_for_line:
        an.arrow_patch.set_color(tkw.get("color", "k"))
    else:
        an.arrow_patch.set_color("k")
    return an
