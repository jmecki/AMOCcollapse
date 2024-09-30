"""Lazy imports for jupyter notebooks, just add this to the top of your
notebook:

%run imports.py or %load imports.py
"""

print("Hi there, have a great and productive day :)")
import optim_esm_tools as oet
import os
import xarray as xr
import numpy as np

import typing as ty
import collections

import matplotlib.pyplot as plt
from immutabledict import immutabledict
import cartopy.crs as ccrs
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

from tqdm.notebook import tqdm

import datetime
import glob
import pandas as pd
from collections import defaultdict
from collections import Counter
import logging
import time
import itertools
import scipy
import sys
import numba
import inspect
import psutil
import shutil
import shlex
import subprocess
import matplotlib as mpl
from functools import partial
from matplotlib.colors import LogNorm, Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from functools import partial, lru_cache
import regionmask
from scipy.signal.windows import gaussian

import scipy.ndimage 
import seawater
from dataclasses import dataclass