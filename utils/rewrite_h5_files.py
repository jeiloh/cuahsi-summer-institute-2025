import os
import datetime
import json
import re
import copy
from scipy.sparse import csr_matrix, lil_matrix, eye, csgraph
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.sparse
import matplotlib.pyplot as plt
import time
from numba import njit

basedir = '/insert/extraction/file/path/here'
outputdir = '/insert/final/export/file/path/here'
num_keys = 1
regex_str = '^([0-9]{10})_([a-z_]+)_([a-zA-z]+).h5$'
regex = re.compile(regex_str)

if __name__ == "__main__":
    for record in os.listdir(basedir):
        print(record)
        if record.startswith('.'):
            continue
        datadir = f'{basedir}/{record}'
        for fn in os.listdir(datadir):
            tables = []
            match = regex.match(fn)
            if match:
                timestamp, series, variable = match.groups()
            for i in range(num_keys):
                hdf = pd.read_hdf(f'{datadir}/{fn}', key=f'data_{i}')
                tables.append(hdf)
            table = pd.concat(tables, axis=1)
            table = table.tz_localize('UTC')
            table = table.sort_index(axis=0).sort_index(axis=1)
            table.to_hdf(f'{outputdir}/{timestamp}.h5', key=f'{series}__{variable}', mode='a')