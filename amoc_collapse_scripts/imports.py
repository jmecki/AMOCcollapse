"""Lazy imports for jupyter notebooks, just add this to the top of your
notebook:

%run imports.py or %load imports.py
"""
import collections
import datetime
import glob
import inspect
import itertools
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
import typing as ty
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from functools import partial

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from scipy.signal.windows import gaussian
from tqdm.notebook import tqdm
