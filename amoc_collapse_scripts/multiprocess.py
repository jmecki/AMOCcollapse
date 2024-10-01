import os
import secrets
import shlex
import subprocess
import time
import typing as ty
from dataclasses import dataclass

import numpy as np
import optim_esm_tools as oet
import psutil


@dataclass
class ResultItem:
    t0: float
    t1: float
    cmds: ty.List[str]
    processes: ty.List[subprocess.Popen]

    @staticmethod
    def perc(x):
        return np.sum(x) / len(x)

    @property
    def succes_rate(self):
        return self.perc([p.returncode == 0 for p in self.processes])

    def __repr__(self):
        dt = self.t1 - self.t0
        time_mess = (
            f"{dt/3600:.1f} h"
            if dt > 3600
            else (f"{dt/60:.1f} m" if dt > 60 else f"{dt:.1f} s")
        )
        return f"Done took {time_mess}, with a {self.succes_rate:.1%} succes_rate"

    def failed_jobs(self):
        return [p for p in self.processes if p.returncode != 0]

    def failed_cmds(self):
        return [" ".join(p.args) for p in self.failed_jobs()]

    def resubmit(self, **kw):
        return submit_jobs(self.failed_cmds(), **kw)


def write_job(job, temp_job_dir, nice=10):
    job_file = os.path.join(temp_job_dir, f"job_{secrets.token_hex(nbytes=16)}.sh")
    with open(job_file, "w") as ff:
        ff.write(job)
    return f"nice -n {nice} bash {job_file}"


def submit_jobs(
    cmds: ty.List[str],
    n_max: int = 15,
    sleep: ty.Union[int, float] = 1,
    n_min: int = 1,
    p_load: ty.Union[bool, float] = False,
    max_mem_for_load: ty.Union[int, float] = 70,
    max_mem_for_start: ty.Union[int, float] = 70,
    break_file: ty.Optional[str] = None,
    write_bash_jobs_in: ty.Optional[str] = None,
    **kw,
) -> ResultItem:
    t0 = time.time()
    processes: ty.List[subprocess.Popen] = []
    if write_bash_jobs_in is not None:
        cmds = [write_job(cmd, write_bash_jobs_in) for cmd in cmds]

    pbar = oet.utils.tqdm(total=len(cmds))

    def n_running(procs):
        procs = [p for p in procs if p.poll() is None]
        return len(procs)

    def _stat(processes):
        n = n_running(processes)
        mem = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent()
        return n, mem, cpu

    # Call once since cpu is time since last cpu call
    _stat([])

    for i, job in enumerate(cmds):
        if break_file is not None and os.path.exists(break_file):
            print("\nbreak\n")
            os.remove(break_file)
            break

        n, mem, cpu = _stat(processes)
        while (mem > max_mem_for_start or n >= n_max) and n > n_min:
            print(
                f"Sleep. Mem {mem:.0f}% CPU {cpu:.0f}%. n={n}" + " " * 10,
                flush=True,
                end="\r",
            )
            pbar.n = max(i - n, 0)
            pbar.desc = "Queue full" if n == n_max else "Resources"
            pbar.display()
            time.sleep(max(sleep, 1))
            n, mem, cpu = _stat(processes)
        if (
            p_load
            and mem <= max_mem_for_load
            and np.random.choice([True, False], p=[p_load, 1 - p_load])
        ):
            job += " --load"
        pbar.desc = "Submit"
        pbar.display()
        print(
            f"Go. Mem {mem:.0f}% CPU {cpu:.0f}%. n={n}" + " " * 50,
            flush=True,
            end="\r",
        )

        pbar.n = max(i - n, 0)
        proc = subprocess.Popen(
            shlex.split(job),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kw,
        )
        processes.append(proc)

    pbar.desc = "Exhaust queue"
    while n := n_running(processes):
        pbar.n = i - n + 1
        time.sleep(sleep)
        print(f"{n} remaining" + " " * 50, flush=True, end="\r")
        pbar.display()
    pbar.n = i + 1
    pbar.desc = "Done"
    pbar.close()
    t1 = time.time()
    dt = t1 - t0
    print(f"Took {dt:.0f} s ({dt/3600:.1f} h)")

    res = ResultItem(t0=t0, t1=t1, processes=processes, cmds=cmds)
    return res
