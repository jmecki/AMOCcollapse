#
import argparse
import glob
import logging
import os
import time
import typing as ty
from dataclasses import dataclass

import gsw
import numpy as np
import optim_esm_tools as oet
import xarray as xr


class ConstantTEOS10:
    alpha = staticmethod(gsw.alpha)
    beta = staticmethod(gsw.beta)
    cp = staticmethod(gsw.cp_t_exact)
    ct = staticmethod(gsw.CT_from_t)
    g = 9.8  # m/s
    rho_0 = 1027  # kg / m3

    @staticmethod
    def q_heat(hfds, dt_seconds):
        return hfds * dt_seconds

    def q_fresh(self, wfo, dt_seconds):
        return wfo * dt_seconds / self.rho_0

    def b_t_surf(self, sos, tos, q_heat, pressure=0):
        ct = self.ct(SA=sos, t=tos, p=pressure)
        alpha = self.alpha(SA=sos, CT=ct, p=pressure)
        rho_0 = self.rho_0
        g = self.g
        cp = self.cp(SA=sos, t=tos, p=pressure)
        return (g * alpha) * q_heat / (rho_0 * cp)

    def b_s_surf(self, sos, tos, q_fresh, pressure=0):
        ct = self.ct(SA=sos, t=tos, p=pressure)
        beta = self.beta(SA=sos, CT=ct, p=pressure)
        g = self.g
        return (g * beta) * sos * q_fresh

    def b_surf_tot(self, sos, tos, wfo, hfds, dt_seconds, pressure=0):
        q_heat = self.q_heat(hfds, dt_seconds)  # J/m2
        q_fresh = self.q_fresh(wfo, dt_seconds)  # m
        return (
            self.b_t_surf(sos=sos, tos=tos, q_heat=q_heat, pressure=pressure),
            self.b_s_surf(
                sos=sos,
                tos=tos,
                q_fresh=q_fresh,
                pressure=pressure,
            ),
        )

    def _s_t_constants(self, sos, tos, pressure):
        ct = self.ct(SA=sos, t=tos, p=pressure)
        alpha = self.alpha(SA=sos, CT=ct, p=pressure)
        beta = self.beta(SA=sos, CT=ct, p=pressure)
        g = self.g
        c_s = beta * g * sos
        c_t = alpha * g * tos
        return c_t, c_s

    def b_and_f_terms(self, sos, tos, wfo, hfds, dt_seconds, pressure=0):
        b_t, b_s = self.b_surf_tot(sos, tos, wfo, hfds, dt_seconds, pressure=pressure)
        c_t, c_s = self._s_t_constants(sos, tos, pressure)
        return (b_t, b_s, c_t, c_s)


def flatten(a):
    return [b for c in a for b in c]


def double_flat(a):
    return flatten(flatten(a))


def find_files(base, experiment_id, source_id):
    files = [
        [
            oet.analyze.find_matches.find_matches(
                base,
                activity_id="*",
                source_id=source_id,
                required_file=False,
                experiment_id=x,
                max_members=None,
                max_versions=None,
                variant_label="*",
            ),
        ]
        for x in ["historical", experiment_id]
    ]
    return files


def build_merges(f, merge_as, exclude=None) -> None:
    exclude = exclude or "/wo/ /vo/".split()
    if os.path.exists(merge_as):
        return
    merge_files = sorted(glob.glob(f"{f}/*.nc"))
    merge_files = [
        f for f in merge_files if all(k not in f for k in [merge_as, *exclude])
    ]
    if not merge_files:
        return
    oet.analyze.pre_process._merge_sources(merge_files, merge_as)


def read_data_sets(
    merged_files,
    intermediate_file,
    variant_label,
    log,
    min_years=2200 - 1850,
):
    datasets = dict()
    for m in oet.utils.tqdm(merged_files):
        if "histor" in m:
            log.info(f"skip {m}")
            continue
        f = os.path.split(m)[0]
        try:
            ds = oet.read_ds(
                f,
                _historical_path=os.path.join(
                    oet.analyze.find_matches.associate_parent(
                        path=f,
                        match_to="historical",
                        strict=True,
                        required_file=intermediate_file,
                    )[0],
                    intermediate_file,
                ),
                pre_proc_kw=dict(_check_duplicate_years=False),
                add_history=True,
                _file_name=intermediate_file,
                max_time=None,
            )
        except Exception as e:
            log.error(f"ran into {e}, continue")
            continue
        if len(ds["time"]) * 12 < min_years:
            log.error(
                f"{f} has max-time {ds.time.values[-1]}, which is below the minimal requirement. Skip.",
            )
            continue
        if ds.variable_id not in datasets and ds.variant_label == variant_label:
            log.warning(f"add {m}")
            datasets[ds.variable_id] = ds
    return datasets


def sign_wfo(ds_wfo):
    avg_wet_region = float(
        ds_wfo["wfo"]
        .isel(time=slice(0, 120))
        .mean(
            "time",
        )
        .sel(lat=slice(10, -10), lon=slice(120, 170))
        .mean("lat lon".split()),
    )

    return int(avg_wet_region > 0) * 2 - 1


def calculate_b_surf(datasets, log, load=False) -> xr.Dataset:
    calc = ConstantTEOS10()
    results = []
    sign = sign_wfo(ds_wfo=datasets["wfo"])
    log.warning(f"this set has sign {sign}!!")
    dt_list = []
    if not load:
        for ts in oet.utils.logged_tqdm(
            range(len(datasets["sos"].time)),
            desc="months",
            log=log,
        ):
            _dt_seconds = (
                np.diff(datasets["sos"]["time_bnds"].values[ts])
                .squeeze()
                .item()
                .total_seconds()
            )
            dt_list.append(_dt_seconds)

            flat_result: np.ndarray = calc.b_and_f_terms(
                sos=datasets["sos"]["sos"].isel(time=ts).values.flatten(),
                tos=datasets["tos"]["tos"].isel(time=ts).values.flatten(),
                wfo=sign * datasets["wfo"]["wfo"].isel(time=ts).values.flatten(),
                hfds=datasets["hfds"]["hfds"].isel(time=ts).values.flatten(),
                dt_seconds=_dt_seconds,
            )
            results.append(flat_result)

    else:
        _dt_seconds = (
            np.diff(datasets["sos"]["time_bnds"].values[0])
            .squeeze()
            .item()
            .total_seconds()
        )
        results = [
            calc.b_and_f_terms(
                sos=datasets["sos"]["sos"].values.flatten(),
                tos=datasets["tos"]["tos"].values.flatten(),
                wfo=sign * datasets["wfo"]["wfo"].values.flatten(),
                hfds=datasets["hfds"]["hfds"].values.flatten(),
                dt_seconds=_dt_seconds,
            ),
        ]

    results_np = np.array(results)

    cell_area = datasets["sos"]["cell_area"].copy().load()
    time_len = len(datasets["sos"]["time"])

    b_t_surf = results_np[:, 0].reshape(time_len, *cell_area.shape)
    b_s_surf = results_np[:, 1].reshape(time_len, *cell_area.shape)

    f_t = results_np[:, 2].reshape(time_len, *cell_area.shape)
    f_s = results_np[:, 3].reshape(time_len, *cell_area.shape)
    buff = datasets["sos"]["sos"]

    b_t_surf_da = buff.copy()
    b_t_surf_da.data = b_t_surf
    b_t_surf_da.name = "b_t_surf"
    b_t_surf_da.attrs.update(
        dict(long_name="$B^T_{surf}$", short_name="b_t_surf_da", units=r"J/kg"),
    )

    b_s_surf_da = buff.copy()
    b_s_surf_da.data = b_s_surf
    b_s_surf_da.name = "b_s_surf"
    b_s_surf_da.attrs.update(
        dict(long_name="$B^S_{surf}$", short_name="b_s_surf_da", units=r"J/kg"),
    )

    f_t_da = buff.copy()
    f_t_da.data = f_t
    f_t_da.name = "f_t"
    f_t_da.attrs.update(dict(long_name="$F^T$", short_name="f_t", units=r"m/s^2"))

    f_s_da = buff.copy()
    f_s_da.data = f_s
    f_s_da.name = "f_s"
    f_s_da.attrs.update(dict(long_name="$F^S$", short_name="f_s", units=r"m/s^2"))

    dt_da = datasets["sos"]["time"].copy()
    dt_da.data = dt_list
    dt_da.attrs = dict(long_name="total seconds", short_name="dt", units="s")

    b_t_surf_da_flux = b_t_surf_da.copy()
    b_t_surf_da_flux = b_t_surf_da / dt_da
    b_t_surf_da_flux.attrs.update(dict(units="W"))
    b_s_surf_da_flux = b_s_surf_da.copy()
    b_s_surf_da_flux = b_s_surf_da / dt_da
    b_s_surf_da_flux.attrs.update(dict(units="W"))
    ds_bsurf_tot = xr.Dataset(
        dict(
            b_t_surf=b_t_surf_da,
            b_s_surf=b_s_surf_da,
            f_t=f_t_da,
            f_s=f_s_da,
            cell_area=cell_area,
            time_bnds=datasets["sos"]["time_bnds"].copy(),
            dt_da=dt_da,
            b_t_surf_da_flux=b_t_surf_da_flux,
            b_s_surf_da_flux=b_s_surf_da_flux,
        ),
        attrs={k: datasets[k].file for k in "sos tos wfo hfds".split()},
    )
    return ds_bsurf_tot


def main(
    base: str,
    experiment_id: str,
    source_id: str,
    intermediate_file: str,
    save_as: str,
    variant_label: str,
    log: logging.Logger,
):
    log.warning("Find files")
    files = double_flat(find_files(base, experiment_id, source_id))
    merged_files = [os.path.join(f, intermediate_file) for f in files]
    for file, merge_as in oet.utils.logged_tqdm(
        list(zip(files, merged_files)),
        log=log,
        desc="Build merges",
    ):
        build_merges(file, merge_as)

    log.warning("Read datasets")
    datasets = read_data_sets(
        merged_files,
        intermediate_file,
        variant_label=variant_label,
        log=log,
    )
    if missing := (set("sos tos hfds wfo".split()) - set(list(datasets))):
        log.critical(f"Missing {missing} break")
        raise ValueError(
            f"Missing required variables for {source_id} {experiment_id} {variant_label}",
        )
    log.warning("Build b-surf")
    ds_bsurf_tot = calculate_b_surf(datasets, log)
    os.makedirs(os.path.split(save_as)[0], exist_ok=True)
    # oet.analyze.pre_process.save_nc(ds_bsurf_tot, save_as)
    ds_bsurf_tot.to_netcdf(save_as)
    log.warning("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build b-surf")
    parser.add_argument(
        "--base",
        default="/data/volume_2/2025_01_14_monthly_amoc_buoyancy/CMIP6/",
        type=str,
    )
    parser.add_argument("--intermediate_file", default="month_merged.nc", type=str)

    parser.add_argument("-e", "--experiment_id", type=str)
    parser.add_argument("-s", "--source_id", type=str)
    parser.add_argument("-v", "--variant_label", type=str)
    parser.add_argument("--save_as", type=str)

    parser.add_argument("--profile_memory", action="store_true")

    logger = oet.get_logger(__name__)

    logger.setLevel(logging.INFO)
    args = parser.parse_args()

    save_as = args.save_as or os.path.join(
        args.base,
        "..",
        "b_surf_calculated",
        f"{args.source_id}_{args.experiment_id}_{args.variant_label}.nc",
    )
    if os.path.exists(save_as):
        logger.error(f"Already done {save_as}")
        import sys

        sys.exit(0)
    main_kw = dict(
        log=logger,
        base=args.base,
        experiment_id=args.experiment_id,
        source_id=args.source_id,
        intermediate_file=args.intermediate_file,
        variant_label=args.variant_label,
        save_as=save_as,
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
