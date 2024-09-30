import json
import os
import sys
import time
import typing as ty
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import optim_esm_tools as oet
import xarray as xr

from . import helper_scripts
from .amoc_deep_dive import _base_mask
from .amoc_deep_dive import set_time_int
from .manual_msft import get_lev_coord

log = oet.get_logger()


def read_config(path_of_config) -> dict:
    with open(path_of_config) as f:
        config = json.load(f)

    return config


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
