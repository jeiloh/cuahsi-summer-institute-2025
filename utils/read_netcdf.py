from __future__ import annotations
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm

import dask as dask
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from shapely.ops import unary_union

from utils import get_nwm_filename

CACHE_DIR = os.environ['NWM_CACHE_DIR']


def extract_timeseries(
    comids: list,
    parameter: str | list[str],
    start_date: datetime,
    days: int = 1,
    rng="short_range",  # analysis_assim
    t0=0,  # 0..23
    product="channel_rt",  # channel_rt, land, reservoir, terrain_rt
    territory="conus",
    mem="1",
    is_old_medium_range=False,
    is_forcing=False,
    timesteps: list[int] = None,
    verbose: bool = False
):
    if isinstance(parameter, str):
        parameter = [parameter]
    # TODO: create nwm_elements
    # get the filenames   #
    #
    if not timesteps:
        if rng == "short_range":
            timesteps = range(1,19)
        elif rng == "medium_range":
            timesteps = range(1,241,1)
        elif rng == "medium_range_blend":
            timesteps = range(1,241,1)
        elif rng == "analysis_assim":
            timesteps = range(0,24)
        elif rng == "analysis_assim_no_da":
            timesteps = range(0,24)

    dt = timedelta(days=1)

    start_time = time.time()
    # open all the corresponding files
    if rng == "short_range":
        filenames = [
            f"{CACHE_DIR}/{get_nwm_filename(t0=t0,product=prod,timestep = f,date=start_date, rng=rng, territory=territory, is_forcing=is_forcing)}"
            for f in timesteps  # all timesteps in the forecast
            for prod in [product]
        ]  # chosen products
    elif rng == "medium_range":
        filenames = [
            f"{CACHE_DIR}/{get_nwm_filename(t0=t0,product=prod,timestep = f,date=start_date, rng=rng, territory=territory, mem=mem, is_old_medium_range=is_old_medium_range, is_forcing=is_forcing)}"
            for f in timesteps  # all timesteps in the forecast
            for prod in [product]
        ]  # chosen products
    elif rng == "medium_range_blend":
        filenames = [
            f"{CACHE_DIR}/{get_nwm_filename(t0=t0,product=prod,timestep = f,date=start_date, rng=rng, territory=territory, is_forcing=is_forcing)}"
            for f in timesteps  # all timesteps in the forecast
            for prod in [product]
        ] 
    elif rng == "analysis_assim":
        filenames = [
            f"{CACHE_DIR}/{get_nwm_filename(t0=t,product=prod,timestep = 0,date=d, rng=rng, territory=territory, is_forcing=is_forcing)}"
            for d in [start_date + dt * x for x in range(days)]  # all the given days
            for t in timesteps  # all the hours in the day
            for prod in [product]
        ]    
    elif rng == "analysis_assim_no_da":
        filenames = [
            f"{CACHE_DIR}/{get_nwm_filename(t0=t,product=prod,timestep = 0,date=d, rng=rng, territory=territory, is_forcing=is_forcing)}"
            for d in [start_date + dt * x for x in range(days)]  # all the given days
            for t in timesteps  # all the hours in the day
            for prod in [product]
        ]  # chosen products
    else:
        tqdm.write(f"ERROR, unsupported range type:{rng}")
    start_time = time.time()

    # keep only the existing files
    filenames_exist = [fname for fname in filenames if os.path.exists(fname)]
    filenames_missing = [fname for fname in filenames if fname not in filenames_exist]
    if len(filenames_missing) > 0:
        tqdm.write("WARNING: Files are missing from cache!")
        tqdm.write("\n".join(filenames_missing))

    with dask.config.set({"array.slicing.split_large_chunks": True}):
        data = xr.open_mfdataset(
            filenames_exist, chunks={"time": 1}, combine="by_coords", coords="minimal"
        )
    if verbose:
        tqdm.write("Read files --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    results = {}
    for param in parameter:
        data_ts = data[param]

        if is_forcing:
            raster_proj = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs"

            inProj = pyproj.Proj("epsg:4326")
            outProj = pyproj.Proj(projparams=raster_proj)

            comids_t = comids.copy()
            for i, id in enumerate(comids):
                comids_t[i]["lat"], comids_t[i]["lon"] = pyproj.transform(
                    inProj, outProj, id["lat"], id["lon"]
                )

            conv = 3600 / 25.4  # unit conversion constant

            ss = [
                data_ts.sel(x=id["lat"], y=id["lon"], method="nearest")
                .to_pandas()
                .rename(id["comid"])
                .mul(conv)
                for id in comids
            ]  # conversion to mmps-1 to inch/h
            df = pd.DataFrame(ss).transpose()
            # tqdm.write(df)
            pd.set_option("display.max_rows", None, "display.max_columns", None)
            # tqdm.write(df)

        else:
            data_features = data_ts.sel(feature_id=[np.int32(comid) for comid in comids])

            if rng == "short_range":
                df = data_features.to_pandas()
            if rng == "medium_range":
                df = data_features.to_pandas()
            elif rng == "medium_range_blend":
                df = data_features.to_pandas()
            elif rng == "analysis_assim":
                df = (
                    data_features.stack(dim_0=("feature_id", "reference_time"))
                    .to_pandas()
                    .stack()
                    .reset_index()
                    .groupby(["time"])
                    .sum(numeric_only=True)
                )
            elif rng == "analysis_assim_no_da":
                df = (
                    data_features.stack(dim_0=("feature_id", "reference_time"))
                    .to_pandas()
                    .stack()
                    .reset_index()
                    .groupby(["time"])
                    .sum(numeric_only=True)
                )
            # clean ts before and after with 0 values
        results[param] = df
    if verbose:
        tqdm.write(f"Process data: {rng} | {param} | {mem} --- {(time.time() - start_time):.2f} seconds ---" )
    if len(results) == 1:
        return results[list(results.keys())[0]]
    else:
        return results


def extract_raster2poly_timeseries(
    polygons: gpd.GeoDataFrame,
    parameter: str,
    start_date: datetime,
    days: int = 1,
    rng="short_range",  # analysis_assim
    t0=0,  # 0..23
    product="channel_rt",  # channel_rt, land, reservoir, terrain_rt
    forecast_horizon=None,
    territory="conus",
    mem="1",
    is_old_medium_range=False,
    is_forcing=True,
    id_col: str = "ID",
):
    # get the filenames
    if not forecast_horizon:
        if rng == "short_range":
            forecast_horizon = 19
        elif rng == "medium_range":
            forecast_horizon = 242

    dt = timedelta(days=1)

    start_time = time.time()
    # open all the corresponding files
    if rng == "short_range":
        filenames = [
            f"{CACHE_DIR}/{get_nwm_filename(t0=t0,product=prod,timestep = f,date=start_date, rng=rng, territory=territory, is_forcing=is_forcing)}"
            for f in range(1, forecast_horizon, 1)  # all timesteps in the forecast
            for prod in [product]
        ]  # chosen products
    elif rng == "medium_range":
        filenames = [
            f"{CACHE_DIR}/{get_nwm_filename(t0=t0,product=prod,timestep = f,date=start_date, rng=rng, territory=territory, mem=mem, is_old_medium_range=is_old_medium_range, is_forcing=is_forcing)}"
            for f in range(3, forecast_horizon, 3)  # all timesteps in the forecast
            for prod in [product]
        ]  # chosen products
    elif rng == "analysis_assim":
        filenames = [
            f"{CACHE_DIR}/{get_nwm_filename(t0=t,product=prod,timestep = 0,date=d, rng=rng, territory=territory, is_forcing=is_forcing)}"
            for d in [start_date + dt * x for x in range(days)]  # all the given days
            for t in range(0, 24)  # all the hours in the day
            for prod in [product]
        ]  # chosen products
    elif rng == "analysis_assim_no_da":
        filenames = [
            f"{CACHE_DIR}/{get_nwm_filename(t0=t,product=prod,timestep = 0,date=d, rng=rng, territory=territory, is_forcing=is_forcing)}"
            for d in [start_date + dt * x for x in range(days)]  # all the given days
            for t in range(0, 24)  # all the hours in the day
            for prod in [product]
        ]
    else:
        tqdm.write(f"ERROR, unsupported range type:{rng}")
    tqdm.write("Filename gen --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    # keep only the existing files
    filenames_exist = [fname for fname in filenames if os.path.exists(fname)]
    filenames_missing = [fname for fname in filenames if fname not in filenames_exist]
    if len(filenames_missing) > 0:
        tqdm.write("WARNING: Files are missing from cache!")
        tqdm.write("\n".join(filenames_missing))

    with dask.config.set({"array.slicing.split_large_chunks": True}):
        data = xr.open_mfdataset(
            filenames_exist, chunks={"time": 1}, combine="by_coords", coords="minimal"
        )

    tqdm.write("Read files --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    tqdm.write(data)
    tqdm.write(data.variables)
    for x in data.variables:
        tqdm.write(x)

    data_ts = data[parameter]

    if is_forcing:
        raster_proj = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs"

        data_ts = data[parameter]
        tqdm.write(data_ts)

        polygons = polygons.to_crs(crs=raster_proj)

        data_ts.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        data_ts.rio.write_crs(raster_proj, inplace=True)
        boundary = gpd.GeoSeries(unary_union(polygons.geometry.to_list()))
        data_ts = data_ts.rio.clip(
            boundary, drop=True
        )  # limit the raster to the boundary of the polygons to extract

        # iterate through the polygons
        ss = [
            data_ts.rio.clip(row.geometry, drop=False)
            .mean(dim=["x", "y"], keep_attrs=True)
            .to_pandas()
            .rename(row[id_col])
            for i, row in polygons.iterrows()
        ]

        df = pd.DataFrame(ss).transpose()  # unit: mms-1
        # tqdm.write(df)
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        tqdm.write(df)

    else:
        raise TypeError("This method can only be applied to raster forcings!")

    tqdm.write("Process data --- %s seconds ---" % (time.time() - start_time))
    return df
