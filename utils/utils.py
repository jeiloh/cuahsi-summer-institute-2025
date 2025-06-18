import os
import os.path
import shutil
from pathlib import Path
from datetime import datetime
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import requests
import tqdm
import xarray as xr

CACHE_DIR = os.environ['NWM_CACHE_DIR']

def get_nwm_output_url(
    baseurl="storage.googleapis.com",
    bucket="national-water-model",
    date: datetime = datetime(2018, 9, 17),
    rng="short_range",  # analysis_assim
    t0=0,
    timestep=0,
    product="channel_rt",
    territory="conus",
    mem="1",
    is_old_medium_range=False,
    is_forcing=False,
):
    if is_forcing:
        if rng == "short_range":
            return (
                f'https://{baseurl}/{bucket}/nwm.{date.strftime("%Y%m%d")}/forcing_{rng}/'
                f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}.f{str(timestep).zfill(3)}.{territory}.nc"
            )
        elif rng in ["medium_range", "long_range"]:
            if is_old_medium_range:
                return (
                    f'https://{baseurl}/{bucket}/nwm.{date.strftime("%Y%m%d")}/forcing_{rng}/'
                    f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}.f{str(timestep).zfill(3)}.{territory}.nc"
                )
            else:
                return (  # TODO: FIX THIS
                    f'https://{baseurl}/{bucket}/nwm.{date.strftime("%Y%m%d")}/{rng}_mem{mem}/'
                    f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}_{mem}.f{str(timestep).zfill(3)}.{territory}.nc"
                )

    else:
        if rng == "analysis_assim":
            return (
                f'https://{baseurl}/{bucket}/nwm.{date.strftime("%Y%m%d")}/{rng}/'
                f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}.tm{str(timestep).zfill(2)}.{territory}.nc"
            )
        elif rng == "analysis_assim_no_da":
            return (
                f'https://{baseurl}/{bucket}/nwm.{date.strftime("%Y%m%d")}/{rng}/'
                f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}.tm{str(timestep).zfill(2)}.{territory}.nc"
            )
        elif rng == "short_range":
            return (
                f'https://{baseurl}/{bucket}/nwm.{date.strftime("%Y%m%d")}/{rng}/'
                f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}.f{str(timestep).zfill(3)}.{territory}.nc"
            )
        elif rng == "medium_range_blend":
            return (
                f'https://{baseurl}/{bucket}/nwm.{date.strftime("%Y%m%d")}/{rng}/'
                f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}.f{str(timestep).zfill(3)}.{territory}.nc"
            )
        elif rng in ["medium_range", "long_range"]:
            if is_old_medium_range:
                return (
                    f'https://{baseurl}/{bucket}/nwm.{date.strftime("%Y%m%d")}/{rng}/'
                    f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}.f{str(timestep).zfill(3)}.{territory}.nc"
                )
            else:
                return (
                    f'https://{baseurl}/{bucket}/nwm.{date.strftime("%Y%m%d")}/{rng}_mem{mem}/'
                    f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}_{mem}.f{str(timestep).zfill(3)}.{territory}.nc"
                )


def get_nwm_filename(
    date: datetime = datetime(2018, 9, 17),
    rng="short_range",  # analysis_assim
    t0=0,
    timestep=0,
    product="channel_rt",
    territory="conus",
    mem="1",
    is_old_medium_range=False,
    is_forcing=False,
):
    if is_forcing:
        if rng == "short_range":
            return (
                f'nwm.{date.strftime("%Y%m%d")}/forcing_{rng}/'
                f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}.f{str(timestep).zfill(3)}.{territory}.nc"
            )
        elif rng in ["medium_range", "long_range"]:
            if is_old_medium_range:
                return (
                    f'nwm.{date.strftime("%Y%m%d")}/forcing_{rng}/'
                    f"nwm.t{str(t0).zfill(2)}z.{rng}.forcing.f{str(timestep).zfill(3)}.{territory}.nc"
                )
            else:
                return (  # TODO: FIX THIS
                    f'nwm.{date.strftime("%Y%m%d")}/{rng}_mem{mem}/'
                    f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}_{mem}.f{str(timestep).zfill(3)}.{territory}.nc"
                )
    else:  # not forcing
        if rng == "analysis_assim":
            return (
                f'nwm.{date.strftime("%Y%m%d")}/{rng}/'
                f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}.tm{str(timestep).zfill(2)}.{territory}.nc"
            )
        elif rng == "analysis_assim_no_da":
            return (
                f'nwm.{date.strftime("%Y%m%d")}/{rng}/'
                f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}.tm{str(timestep).zfill(2)}.{territory}.nc"
            )
        elif rng == "short_range":
            return (
                f'nwm.{date.strftime("%Y%m%d")}/{rng}/'
                f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}.f{str(timestep).zfill(3)}.{territory}.nc"
            )
        elif rng == "medium_range_blend":
            return (
                f'nwm.{date.strftime("%Y%m%d")}/{rng}/'
                f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}.f{str(timestep).zfill(3)}.{territory}.nc"
            )
        elif rng in ["medium_range", "long_range"]:
            if is_old_medium_range:
                return (
                    f'nwm.{date.strftime("%Y%m%d")}/{rng}/'
                    f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}.f{str(timestep).zfill(3)}.{territory}.nc"
                )
            else:
                return (
                    f'nwm.{date.strftime("%Y%m%d")}/{rng}_mem{mem}/'
                    f"nwm.t{str(t0).zfill(2)}z.{rng}.{product}_{mem}.f{str(timestep).zfill(3)}.{territory}.nc"
                )


def clear(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        return

def get_filename(url) -> Path:
    spliturl = url.split("/")
    if not os.path.exists(f"{CACHE_DIR}/{spliturl[-3]}/{spliturl[-2]}"):
        os.makedirs(f"{CACHE_DIR}/{spliturl[-3]}/{spliturl[-2]}")
    file_name = Path(f"{CACHE_DIR}/{spliturl[-3]}/{spliturl[-2]}/{spliturl[-1]}")
    return file_name


def remove_url_params(url: str) -> str:
    if "?" in url:
        url_without_params = url.split("?")[0]
    else:
        url_without_params = url
    return url_without_params


def download_url(url):
    file_name = get_filename(remove_url_params(url))
    file_name.parent.mkdir(parents=True, exist_ok=True)

    if not file_name.is_file():
        response = requests.get(url, stream=True)
        if response.status_code == requests.codes.ok:
            with open(file_name, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            # print(f"Downloaded:{url} to {file_name}")
            return f"Downloaded:{url} to {file_name}"
        else:
            return f"Retrieve url failed {response.status_code}: {url}"
    else:
        return f"Already exist:{file_name}"


def download_mp(url_list, n_process=cpu_count()):
    from functools import partial
    from multiprocessing import Pool

    # filter existing files
    fil_urls = [u for u in url_list if not get_filename(u).exists()]
    print(f"{len(url_list)-len(fil_urls)}/{len(url_list)} already exists downloading {len(fil_urls)} files")

    num_files = len(fil_urls)
    num_existing_files = len(url_list) - num_files

    if len(fil_urls) > 0:
        pool = Pool(n_process)
        download_func = partial(download_url)
        for _ in tqdm.tqdm(
            pool.imap_unordered(download_func, fil_urls), total=num_files
        ):
            pass
        pool.close()
        pool.join()
    else:
        print("All the requested files are cached.")

# import concurrent.futures 
# def download_mt(url_list, n_process=cpu_count()):
#     with concurrent.futures.ThreadPoolExecutor() as executor: 
#         results = list(tqdm.tqdm(executor.map(download_url, url_list), total=len(url_list)))
#     return results
        


def download_sequential(url_list):
    return [download_url(url=url) for url in url_list]


def df_to_zrxp(
    df: pd.DataFrame,
    loc_exid: str,
    ts_exid: str,
    ts_name: str,
    separator: str = "|*|",
    dir="temp/",
    extension=".zrxp",
    chunksize=None,
    header: str = None,
    **kwargs,
):
    _kwargs = {
        "SANR": loc_exid,
        "REXCHANGE": ts_exid,
        "ts_name": ts_name,
        "CUNIT": "m3 s-1",
        "ts_spacing": "PT1H",
        "TZ": "UTC-5",
        "LAYOUT": "(timestamp,value)",
        "ZRXPCREATOR": "nwm_ts_script",
        **kwargs,
    }
    if not header:
        header = (
            "#" + separator.join([f"{key}{value}" for key, value in _kwargs.items()]) + "\n"
        )

    if chunksize:
        list_df = [df[i : i + chunksize] for i in range(0, df.shape[0], chunksize)]
    else:
        list_df = [df]

    filenamelist = []

    for i, subdf in enumerate(list_df):
        if not os.path.exists(dir):
            os.makedirs(dir)
        if chunksize:
            filename = f"{dir}{ts_exid}_{str(i).zfill(3)}{extension}"
        else:
            filename = f"{dir}{ts_exid}{extension}"
        f = open(filename, "w+")
        f.write(header)

        for index, value in subdf.iteritems():
            time = index.strftime("%Y%m%d%H%M%S")
            f.write(f"{time}\t{value:.9f}\n")

        filenamelist.append(filename)

    return filenamelist


def execute_ds_cli(cmd_string: str):
    os.system(cmd_string)


def execute_mp(cmd_list, n_process=cpu_count()):
    from functools import partial
    from multiprocessing import Pool

    pool = Pool(n_process)

    func = partial(execute_ds_cli)
    # pool.imap_unordered(func, cmd_list)
    for _ in tqdm.tqdm(pool.imap_unordered(func, cmd_list), total=len(cmd_list)):
        pass
    pool.close()
    pool.join()


def read_xsecs(filename: str, vars: list = None, index_col: str = None) -> pd.DataFrame:
    int32_to_int64 = True

    xar = xr.open_dataset(filename)
    for varname, da in xar.data_vars.items():
        print(f"'{varname}':{da.attrs}\n")

    df1 = xar["streamflow"].to_dataframe()
    print(df1)

    if vars is not None:
        df = xar["link"].to_dataframe()
        for var in vars:
            df[var] = xar[var].data
    else:
        df = xar.to_dataframe()

    if index_col is not None:
        df = df.set_index(index_col)
    if int32_to_int64:
        d = dict.fromkeys(df.select_dtypes(np.int32).columns, np.int64)
        df = df.astype(d)

    return df


def read_ts_with_cli():
    pass


def ds_io(io_mapping):
    pass
