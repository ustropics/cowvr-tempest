import datetime
import earthaccess
import glob
import h5py
import logging
import multiprocessing
import os
import shutil

import sys
import xarray
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from requests.exceptions import HTTPError

# Custom filter to exclude INFO level messages


class ExcludeInfoFilter(logging.Filter):
    def filter(self, record):
        # Allow DEBUG, WARNING, ERROR, and CRITICAL; exclude INFO
        return record.levelno != logging.INFO
