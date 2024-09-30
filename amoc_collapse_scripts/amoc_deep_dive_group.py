import json
import os
import sys
import time
import typing as ty
from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


from .import amoc_deep_dive
import optim_esm_tools as oet
from . import helper_scripts
from .amoc_deep_dive import _base_mask
from .amoc_deep_dive import get_lev_coord
from .amoc_deep_dive import plot_and_modify_buffer
from .amoc_deep_dive import plot_mlotst_difference
from .amoc_deep_dive import PlotProducer
from .amoc_deep_dive import set_time_int
from .amoc_deep_dive import update_units
from .amoc_deep_dive import yearly_average

log = oet.get_logger()



def read_config(path_of_config) -> dict:
    with open(path_of_config) as f:
        config = json.load(f)

    return config


def reference(key: str) -> int:
    vals = {
        "ACCESS-CM2": 800,
        "CanESM5": 300,
        "CESM2-WACCM": 600,
        "GISS": 1300,
        "IPSL": 300,
        "MRI": 800,
        "UKESM1-0": 500,
    }
    for k, v in vals.items():
        if k in key:
            return v
    return 200


def round_range(down, up, frac=0.025):
    span = np.abs(down - up)
    dy = span * frac
    return down - dy, up + dy


def plot_zones_grouped(read_paths):
    for suptitle in oet.utils.tqdm(read_paths):
        log.info(f"Skip {suptitle}")
        continue


# def bla():
#         folders = {k.split('_')[1]: v for k, v in read_paths[suptitle].items() if 'folder_' in k}
#         folder_ds = {k: oet.read_ds(v, add_history=True, max_time=None) for k,v in folders.items() if v is not None}

#         ds = oet.read_ds(folders['mlotst'], add_history=True, max_time=None)
#         reference_masks = amoc_deep_dive.plot_mlotst_cells(
#             ds=ds,
#             label=suptitle,
#             field=f"{ds.variable_id}_run_mean_10",
#             reference_depth=reference(id_key),
#             smooth_reference=True,
#             min_cells=15,
#             save_kw=dict(
#                 name=f"mlotst_convection",
#                 sub_dir=id_key,
#                 **fig_kw,
#             ),
#         )
#         for variable, ds_var in folder_ds.items():
#             set_time_int(ds_var)

#             for k, s in reference_masks.items():
#                 ds_reg = ds_var.where(_base_mask(ds_var, hemisphere='north'), drop=True).where(s, drop=False)
#                 y = helper_scripts.weighted_mean_array(ds_reg, field=f'{ds_reg.variable_id}_run_mean_10')
#                 plt.plot(ds_reg["time"], y, label=f"Reg. {k} of {suptitle}")


#             plt.title('')
#             plt.legend(**oet.utils.legend_kw(ncol=1))
#             plt.ylabel(f'$\\mathrm{{rm}}_{{10}}$ {oet.plotting.plot.get_ylabel(ds)}')
#             plt.show()
#         for reference_depth in oet.utils.tqdm(np.arange(100, 2501, 100)):
#             amoc_deep_dive.plot_mlotst_cells(
#                 ds=ds,
#                 label=suptitle,
#                 field=f"{ds.variable_id}_run_mean_10",
#                 reference_depth=reference_depth,
#                 smooth_reference=True,
#                 min_cells=15,
#                 save_kw=dict(
#                     name=f"{reference_depth:04}_mlotst_convection",
#                     sub_dir=f"{id_key}/references",
#                     **fig_kw,
#                 ),
# )
def _label_to_id(suptitle):
    id_key = suptitle.replace(" ", "_")
    return id_key


def plot_all_refence_depths_mlotst(config):
    for suptitle, path_dict in read_config(config["read_from"]).items():
        if path_dict.get("folder_mlotst") is None:
            continue
        for reference_depth in oet.utils.tqdm(
            np.arange(100, 2501, 100),
            desc=f"Depths of {suptitle}",
        ):
            ds = oet.read_ds(
                path_dict.get("folder_mlotst"),
                max_time=None,
                add_history=True,
            )
            amoc_deep_dive.plot_mlotst_cells(
                ds=ds,
                label=suptitle,
                field=f"{ds.variable_id}_run_mean_10",
                reference_depth=reference_depth,
                smooth_reference=True,
                min_cells=15,
                save_kw=dict(
                    name=f"{reference_depth:04}_mlotst_convection",
                    sub_dir=f"{_label_to_id(suptitle)}/references",
                    **config.get("fig_kw", {}),
                ),
            )


def find_mlotst_reference_regions(
    config: ty.Dict,
) -> ty.Dict[str, ty.Dict[str, ty.List[np.ndarray]]]:
    folder_dict: ty.Dict[str, str] = read_config(config["read_from"])
    mlotst_masks: ty.Dict[str, ty.Dict[str, ty.List[np.ndarray]]] = {}

    for label in oet.utils.tqdm(folder_dict, desc="loop over labels"):
        if config.get("keep_labels") and label not in config["keep_labels"]:
            log.info(f"Skip {label} since it's said so by the config")
            continue
        ds_mlotst = oet.read_ds(
            folder_dict[label]["folder_mlotst"],
            add_history=True,
            max_time=None,
        )
        res = amoc_deep_dive.plot_mlotst_cells(
            ds_mlotst,
            reference_depth=reference(label),
            min_cells=15,
        )
        mlotst_masks[label] = res
        log.info(f"Have regions {(list(res))} for {label} at {reference(label)} m")
        # res2 = plot_mlotst_cells(ds_mlotst, reference_depth=450, min_cells=15, show=False)
        # use = {**{f'{k}$>$1300m': v for k,v in res.items()},
        #        **{'C (by hand)': da.where(_base_mask(ds_mlotst), drop=True).astype(bool)}}
        ma_group = res["A"].copy().astype(int)
        for i, g in enumerate(mlotst_masks[label].values()):
            ma_group.data[g.values] = i + 1

        plot_and_modify_buffer(
            ma_group,
            smooth=False,
            contours=False,
            add_colorbar=False,
            cmap="Blues",
        )
        for k, ss in mlotst_masks[label].items():
            y, x = amoc_deep_dive.median_lat_lon(ss, ds_mlotst)
            plt.text(x, y, k, c="r", ha="center", va="center")
        plt.title(f"Regions {label} based on mlotst")
        oet.utils.save_fig(
            name=f"mlotst_convection_ROI",
            # sub_dir=os.path.join('compare_groups', label.replace(' ', '_')),
            sub_dir=os.path.join(_label_to_id(label)),
            **config.get("fig_kw", {}),
        )
        oet.plotting.plot._show(False)
    return mlotst_masks


def time_series_in_regions(config: ty.Dict, mlotst_masks=None):
    mlotst_masks = mlotst_masks or find_mlotst_reference_regions(config)
    folder_dict: ty.Dict[str, dict] = read_config(config["read_from"])
    variables = sorted(
        {
            k1
            for v0 in folder_dict.values()
            for k1, v1 in v0.items()
            if v1 is not None and "folder_" in k1
        },
    )

    for variable_folder in oet.utils.tqdm(variables, desc="looping over variables"):
        time_series = defaultdict(dict)

        log.info(f"Making {variable_folder}")
        for label, use in mlotst_masks.items():
            path = folder_dict[label].get(variable_folder)
            if path is None:
                log.info(f"Skipping {variable_folder} for {label}")
                continue

            ds_var = oet.read_ds(path, add_history=True, max_time=None)
            set_time_int(ds_var)

            for k, s in use.items():
                ds_reg = ds_var.where(_base_mask(ds_var), drop=True).where(
                    s,
                    drop=False,
                )
                y = helper_scripts.weighted_mean_array(
                    ds_reg,
                    field=f"{ds_reg.variable_id}_run_mean_10",
                )
                time_series[label][k] = dict(time=ds_reg["time"], y=y)
        if not time_series:
            log.info(f"Skipping {variable_folder}")
            continue
        for label, v in time_series.items():
            for k, doc in v.items():
                plt.plot(doc["time"], doc["y"], label=f"Reg. {k} of {label}")
            plt.title("")
            plt.legend(**oet.utils.legend_kw(ncol=1))
            plt.ylabel(
                f"$\\mathrm{{rm}}_{{10}}$ {oet.plotting.plot.get_ylabel(ds_var)}",
            )
            all_val = [y["y"] for yy in time_series.values() for y in yy.values()]
            all_val = [aa for a in all_val for aa in a]
            try:
                plt.ylim(*round_range(np.nanmin(all_val), np.nanmax(all_val)))
            except ValueError:
                for k in "all_val label time_series v":
                    print(f"{k} = {eval(k)}")
                raise
            oet.utils.save_fig(
                name=f'{variable_folder.replace("folder_", "")}_time_series',
                sub_dir=os.path.join("compare_groups", label.replace(" ", "_")),
                **config.get("fig_kw", {}),
            )
            oet.plotting.plot._show(False)

        _all_val_among_members = []
        for label, v in time_series.items():
            for k, doc in v.items():
                plt.plot(doc["time"], doc["y"], label=f"Reg. {k} of {label}")
            all_val = [y["y"] for yy in time_series.values() for y in yy.values()]
            all_val = [aa for a in all_val for aa in a]
            _all_val_among_members += all_val
        plt.title("")
        plt.legend(**oet.utils.legend_kw(ncol=1))
        plt.ylabel(f"$\\mathrm{{rm}}_{{10}}$ {oet.plotting.plot.get_ylabel(ds_var)}")

        try:
            plt.ylim(
                *round_range(
                    np.nanmin(_all_val_among_members),
                    np.nanmax(_all_val_among_members),
                ),
            )
            plt.ylim(*round_range(np.nanmin(all_val), np.nanmax(all_val)))
        except ValueError:
            for k in "_all_val_among_members label time_series v".split():
                print(f"{k} = {eval(k)}")
            raise
        oet.utils.save_fig(
            name=f'{variable_folder.replace("folder_", "")}_time_series',
            sub_dir=os.path.join(
                "compare_groups",
                "multi",
                config.get("group_label", "group"),
            ),
            **config.get("fig_kw", {}),
        )
        oet.plotting.plot._show(False)


@dataclass
class WaterBudget:
    variables: ty.List[str]
    labels: ty.List[str]
    y_vals: ty.List[np.ndarray]
    y_offset: ty.List[np.ndarray]
    timestamps: ty.List[np.ndarray]
    total_sum: np.ndarray
    reference_slice: slice
    reference_period: ty.Tuple[int, int]
    label_y: str
    label_dy: str


def ts_budget(*a, **kw) -> np.ndarray:
    return ts_budget_inner(*a, **kw).total_sum


def ts_budget_inner(
    ds_dict,
    mask=None,
    rm=10,
    average_slice=slice(5, 15),
    variables=None,
    minus_variables=None,
    skip_sum=None,
    _report_missing=True,
    add_sum_to_label=True,
    add_minus_wfo=True,
    update_labels=True,
    show_sum=False,
    split_ax=True,
) -> WaterBudget:
    if split_ax:
        _, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(10, 10),
            gridspec_kw=dict(hspace=0.02),
            dpi=50,
        )
    else:
        plt.figure()
        axes = [plt.gca(), plt.gca()]
    variables = variables or "wfo pr evs friver fsitherm (1-siconc)*pr ".split()
    minus_variables = minus_variables or "evs".split()
    skip_sum = skip_sum or "wfo pr".split()
    ds_siconc = ds_dict["siconc"]
    if "(1-siconc)*pr" in variables:
        ds_dict["(1-siconc)*pr"] = xr.Dataset(
            {
                "(1-siconc)*pr": (1 - ds_siconc[ds_siconc.variable_id] / 100)
                * ds_dict["pr"]["pr"],
                "cell_area": ds_dict["siconc"]["cell_area"],
            },
            attrs=dict(variable_id="(1-siconc)*pr"),
        )
    for v in variables:
        if v not in ds_dict:
            log.error(f"Missing {v}")

    if mask is None:
        ref_var = "wfo" if "wfo" in ds_dict else "sos"
        mask = _base_mask(ds_dict[ref_var]) & ~np.isnan(
            ds_dict[ref_var][ref_var].isel(time=0),
        )

    _sum = None

    res_labels = []
    res_y_vals = []
    res_y_offset = []
    res_timestamps = []
    for v in variables:
        a = ds_dict[v].where(mask, drop=False)
        y = helper_scripts.weighted_mean_array(a, field=v)
        y = -y if v in minus_variables else y
        y_rm = helper_scripts.running_mean(y, rm)
        y_offset = np.nanmean(y[average_slice])
        v_label = (
            v if v != "(1-siconc)*pr" else "$(1-\\mathrm{siconc}) \\times \\mathrm{pr}$"
        )
        v_label = (
            v_label if v_label not in minus_variables else rf"$-\mathrm{{{v_label}}}$"
        )

        if v not in skip_sum and add_sum_to_label:
            v_label += r"$^\Sigma$"
        x = a["time"]

        kw = dict(label=v_label)
        if _sum is None:
            _sum = y * 0

        _ly, _ls = len(y), len(_sum)
        if _ly != _ls:
            if _ly > _ls:
                _sum = np.concatenate([_sum, np.array([np.nan] * (_ly - _ls))])
            else:
                y = np.concatenate([y, np.array([np.nan] * (_ls - _ly))])
        if add_minus_wfo and v == "wfo":
            _sum -= y
        elif v in skip_sum:
            ...
        else:
            _sum += y
        if split_ax:
            axes[0].plot(x, y, **kw)
            kw.pop("label")

        axes[1].plot(x, y_rm - y_offset, **kw)
        res_labels.append(v_label if update_labels else v)
        res_y_vals.append(y)
        res_y_offset.append(y_offset)
        res_timestamps.append(x)

    if show_sum:
        kw_sum = dict(ls="dotted", label=r"$\Sigma$", c="k", lw=1)
        if add_minus_wfo:
            kw_sum["label"] = "Residual"
        if split_ax:
            axes[0].plot(x, _sum, **kw_sum)
        _dy_sum = helper_scripts.running_mean(_sum, rm) - np.nanmean(
            helper_scripts.running_mean(_sum, rm)[average_slice],
        )
        kw_sum.pop("label")
        axes[1].plot(x, _dy_sum, **kw_sum)

    axes[0].legend(**oet.utils.legend_kw(ncol=4, handlelength=0.75))
    if "kg" in a.attrs.get("units", "kg m-2 s-1"):
        y_lab = "Water flux (into seawater) [$10^{-5}$ kg m$^{-2}$ s$^{-1}$]"
        dy_lab = f"$\\Delta$Water flux $\\mathrm{{rm}}_{{{rm}}}$ [$10^{{-5}}$ kg m$^{{-2}}$ s$^{{-1}}$]"
    elif "mm/year" in a.attrs.get("units", "kg m-2 s-1"):
        y_lab = "Water flux (into seawater) [mm/year]"
        dy_lab = f"$\\Delta$Water flux $\\mathrm{{rm}}_{{{rm}}}$ [mm/year]"
    axes[0].set_ylabel(y_lab)
    axes[1].set_ylabel(dy_lab)
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    x0 = x.values[average_slice][0]
    x1 = x.values[average_slice][-1]
    axes[1].axvspan(
        x0,
        x1,
        alpha=0.1,
        color="k",
        label="Reference period",
    )
    axes[1].legend(
        **oet.utils.legend_kw(
            bbox_to_anchor=None,
            loc="upper left",
            ncol=1,
            mode=None,
            borderaxespad=None,
            handlelength=0.75,
        ),
    )

    missing = set("wfo pr evs friver fsitherm (1-siconc)*pr".split()) - set(variables)
    if missing and _report_missing:
        plt.text(
            1850,
            plt.ylim()[0] + 0.25,
            f'Missing {", ".join(missing)}',
            bbox=dict(facecolor="gainsboro", edgecolor="black", boxstyle="round"),
        )
    return WaterBudget(
        variables=variables,
        labels=res_labels,
        y_vals=res_y_vals,
        y_offset=res_y_offset,
        total_sum=_sum,
        timestamps=res_timestamps,
        reference_slice=average_slice,
        reference_period=(x0, x1),
        label_y=y_lab,
        label_dy=dy_lab,
    )


def plot_budget(config, mlotst_masks=None):
    mlotst_masks = mlotst_masks or find_mlotst_reference_regions(config)
    folder_dict: ty.Dict[str, str] = read_config(config["read_from"])

    for label, mask_dict in mlotst_masks.items():
        mask_dict = {f"Reg. {k}": v for k, v in mask_dict.items()}
        mask_dict["North Atlantic"] = None
        log.info(f"Plot water budget in {label}")
        ds_dict = {
            k.split("_")[1]: set_time_int(
                oet.read_ds(v, add_history=True, max_time=None),
            )
            for k, v in folder_dict[label].items()
            if "folder_" in k and v is not None
        }

        for mask_name, mask in mask_dict.items():
            log.info(f"Plot water budget in {label} {mask_name}")
            ts_budget(
                ds_dict,
                mask=mask,
                average_slice=slice(90, 110),
                rm=50,
                **config.get("ts_kw", {}).get(label, {}),
            )
            plt.suptitle(f"Water budget {label}, {mask_name}", y=1.0)
            _id = _label_to_id(label)
            oet.utils.save_fig(
                name=f'water_budget_{_id}_{mask_name.replace(" ", "_").replace(".", "")}',
                sub_dir=os.path.join(_id),
                **config.get("fig_kw", {}),
            )
            oet.plotting.plot._show(False)


if __name__ == "__main__":
    log.warning("start with main")
    t0 = time.time()
    oet.utils.setup_plt()
    config = read_config(sys.argv[1])
    import logging

    log.setLevel(logging.INFO)

    log.info("read configs")
    oet.config.config.read_dict(config.get("config_update", {}))
    config["fig_kw"] = {
        **dict(file_types=("png",), dpi=100),
        **config.get("fig_kw", {}),
    }

    log.info("Find masks")
    mlotst_masks = find_mlotst_reference_regions(config)

    log.info("Make water-budget")
    plot_budget(config, mlotst_masks=mlotst_masks)

    log.info("Make reference plots")
    time_series_in_regions(config, mlotst_masks=mlotst_masks)

    log.info("Make depth record plots")
    plot_all_refence_depths_mlotst(config)

    log.warning(f"done in {time.time()-t0:.1f}s")
