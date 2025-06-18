import os
# Specify path for storing cache here
CACHE_DIR = "/insert/cache/file/path/here"
os.environ['USE_PYGEOS'] = '0'
os.environ['NWM_CACHE_DIR'] = CACHE_DIR 

import multiprocessing
from functools import partial
import shutil
import json
import time
from copy import copy
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from utils import download_url, get_nwm_output_url, get_filename
from read_netcdf import extract_timeseries

# Specify path for storing output here
EXPORT_DIR = "/insert/extraction/file/path/here"
CACHED_DAYS = 1
NUM_PROCESSES = 8
DL_PROCESSES = 6
START_DATE = datetime(2022, 8, 22)
NUMDAYS = 2
DT = timedelta(days=1)
DH = timedelta(hours=1)
# Specify COMIDS for NWM reaches needed here
COMIDS = [5671187, 5671185, 5671181, 5671189, 5671619, 5671143, 5671117, 5671065, 5671049, 5671593, 5671589, 5671169, 5671153, 5671151, 5671735, 5671119, 5671607, 5671061, 5671701, 5671585, 5671139, 5671149, 5671731, 5671113, 5671603, 5671071, 5671713, 5671079, 5671177, 5671741, 5671163, 5671063, 5671705, 5671053, 5671595, 5671095, 5671055, 5671703, 5671103, 5671125, 5671725, 5671605, 5671155, 5671173, 5671159, 5671147, 5671621, 5671753, 5671219, 5672881, 5673175, 5673171, 5672901, 5673173, 5672885, 5672891, 5672883, 5671221, 5671217, 5672889, 5672909, 5673155, 5673179, 5673157, 5673153, 5673151, 5673177, 5673159, 5671191, 5674167, 5672895, 5671213, 5672903, 5671215, 5671697, 5671695, 5671693, 5671749, 5671747, 5671745, 5671739, 5671737, 5781285, 5781257, 5781235, 5781217, 5781167, 5781155, 5781143, 5781135, 5781127, 5781329, 5781917, 5781919, 5781193, 5781183, 5781189, 5781173, 5781133, 5781163, 5781131, 5781141, 5781157, 5781161, 5779429, 5779425, 5779423, 5779417, 5779401, 5779389, 5779385, 5779381, 5779375, 5779033, 5779021, 5779003, 5778987, 5778973, 5778961, 5778945, 5778909, 5778911, 5779981, 5779983, 5779985, 5781711, 5781333, 5781337, 5781713, 5781325, 5781269, 5781265, 5781267, 5781277, 5781273, 5781297, 5781299, 5781287, 5781293, 5781315, 5780359, 5781327, 5780043, 5780041, 5780039, 5780045, 5781385, 5781931, 5781371, 5781369, 5781375, 5781733, 5781395, 5781403, 5781409, 5781407, 5781373, 5781351, 5781927, 5781357, 5781411, 5781421, 5781939, 5781423, 5781425, 5781417, 5781389, 5781387, 5781363, 5781431, 5781477, 5781481, 5781479, 5781475, 5781473, 5781471, 5781487, 5780087, 5781453, 5780081, 5780069, 5780073, 5780075, 5780065, 5780055, 5780053, 5780061, 5780059, 5780363, 5780361, 5780097, 5781311, 5781719, 5781261, 5781253, 5780029, 5781255, 5780079, 5780089, 5781301, 5780035, 5780037, 5781275, 5781303, 5780091, 5780071, 5781211, 5781313, 5781485, 5780101, 5779031, 5781153, 5781113, 5781759, 5781393, 5781349, 5781925, 5781419, 5781259, 5781203, 5781379, 5781427, 5781437, 5781941, 5781935, 5781937, 5781367, 5781361, 5781731, 5781343, 5781345, 5781359, 5781341, 5781353, 5781397, 5781355, 5781377, 5781391, 5781401, 5781399, 5781383, 5781365, 5781933, 5781381, 5781405, 5781429, 5781435, 5781449, 5781445, 5781415, 5781439, 5781441, 5781457, 5781469, 5781465, 5781467, 5781463, 5781451, 5781945, 5781443, 5780063, 5780067, 5780077, 5780057, 5780085, 5780093, 5780083, 5781489, 5781433, 5781461, 5781447, 5781947, 5781459, 5781953, 5781413, 5781347, 5781335, 5781921, 5781961, 5781729, 5781727, 5781709, 5781247, 5781249, 5781949, 5781271, 5781279, 5781283, 5780031, 5780033, 5780047, 5780049, 5780051, 5781725, 5781951, 5781723, 5781721, 5781717, 5781323, 5781317, 5781715, 5781339, 5781331, 5781289, 5781263, 5781251, 5781229, 5781219, 5781225, 5781207, 5781231, 5781227, 5781245, 5781239, 5781241, 5781213, 5781197, 5781181, 5781179, 5781177, 5781169, 5781793, 5781959, 5780355, 5780023, 5780027, 5780025, 5780353, 5779991, 5779993, 5779261, 5779041, 5779965, 5779967, 5779029, 5779251, 5779253, 5779247, 5779297, 5778997, 5778977, 5778991, 5778979, 5778993, 5779263, 5778953, 5779013, 5778983, 5778959, 5778939, 5778913, 5778975, 5779005, 5779023, 5781099, 5781107, 5781111, 5781109, 5781105, 5781095, 5779421, 5779405, 5779395, 5779413, 5779393, 5779397, 5779409, 5779373, 5779383, 5779377, 5779371, 5778967, 5779367, 5779363, 5779369, 5779365, 5778951, 5779379, 5779387, 5779391, 5779399, 5779403, 5779427, 5779407, 5779411, 5779415, 5779419, 5781101, 5781097, 5781103, 5781121, 5781123, 5781115, 5781117, 5781151, 5781175, 5781205, 5781195, 5781199, 5781171, 5781159, 5781129, 5781139, 5781149, 5781145, 5781781, 5781191, 5781185, 5781187, 5781209, 5781233, 5781243, 5781861, 5781863, 5781703, 5781223, 5781221, 5781897, 5781295, 5781307, 5781319, 5781291, 5781309, 5781281, 5781237, 5781165, 5781119, 5781761, 5781125, 5781137, 5781147, 5781215, 5781201, 5779249, 5779973, 5780417, 5780349, 5781699, 5780095, 5781455, 5781305, 5780351, 5781483, 5780099, 5781901, 5781963, 5781929, 5781923, 24670162, 5781899, 24670164, 5781965, 5781789, 5781791, 5781795, 5781797, 5781807, 5781811, 5781815, 5781819, 5781823, 5781841, 5781839, 5781837, 5781835, 5781833, 5781831, 5781829, 5781827, 5781825, 5781879, 5781877, 5781875, 5781873, 5781871, 5781869, 5781867, 5781865, 5781859, 5781857, 5781855, 5781853, 5781851, 5781849, 5781847, 5781845, 5781843, 5781915, 5781913, 5781911, 5781909, 5781907, 5781905, 5781903, 5781893, 5781891, 5781889, 5781887, 5781885, 5781883, 5781881, 5781821, 5781817, 5781813, 5781809, 5781805, 5781803, 5781801, 5779319, 5779699, 5780373, 5780379, 5781957, 5781955, 5781799, 5781787, 5781785, 5781783, 5781779, 5781777, 5781775, 5781773, 5781771, 5781769, 5781767, 5781765, 5781763, 5781757, 5781755, 5781753, 5781751, 5781749, 5781747, 5781745, 5781743, 5781741, 5781739, 5781737, 5781735, 5780415, 5780411, 5780409, 5780391, 5780387, 5780385, 5780383, 5780381, 5780375, 5780369, 5779325, 5779321, 5780389, 5780377, 5780371, 5780367, 5780365, 5779329, 5779327, 5779323, 5779317, 5779315, 5779313, 5779311, 5779309, 5779307, 5779305, 5779301, 5782719, 5780419, 5789878, 5785239, 5785187, 5785899, 5786009, 5786031, 5785351, 5786029, 5786011, 5785409, 5785153, 5785893, 5785143, 5785281, 5786139, 5785245, 5786145, 5785313, 5785987, 5785359, 5785317, 5785387, 5786033, 5786035, 5785379, 5785407, 5785399, 5786081, 5785425, 5785155, 5785177, 5785181, 5785165, 5785895, 5785901, 5785145, 24670292, 5785985, 5785981, 5780393, 5780395, 5780397, 5780399, 5780403, 5780407, 5780413, 5780405, 5780401, 5789058, 5789044, 5789038, 5789032, 5789020, 5789782, 5788990, 5788982, 5788938, 5788930, 5789766, 5788922, 5788902, 5788872, 5788866, 5788860, 5789752, 5789022, 5789794, 5788976, 5788952, 5788904, 5788882, 5788862, 5788804, 5789954, 5789104, 5789812, 5789122, 5789128, 5789176, 5789178, 5789268, 5789832, 5789232, 5789204, 5789906, 5789336, 5789328, 5789306, 5789834, 5789836, 5789318, 5789314, 5789320, 5789346, 5789378, 5789392, 5789408, 5789412, 5789420, 5789422, 5789430, 5789500, 5789788, 5789796, 5788984, 5789016, 5789872, 5788958, 5789800, 5789012, 5789544, 5788934, 5788896, 5788884, 5788844, 5789830, 5789840, 5789396, 5789282, 5789258, 5789308, 5789322, 5789998, 5789342, 5789388, 5789414, 5790012, 5789406, 5789434, 5789506, 5789440, 5789394, 5789400, 5789124, 5789182, 5789256, 5789172, 5789162, 5789814, 5789824, 5789144, 5789138, 5788978, 5788864, 5788836, 5789852, 5788846, 5788906, 5788954, 5788924, 5789798, 5789010, 5789792, 5788972, 5789862, 5789790, 5789054, 5789066, 5789880, 5789060, 5789056, 5789036, 5789030, 5789876, 5789018, 5789874, 5789780, 5788988, 5788980, 5788898, 5788928, 5788888, 5788886, 5788834, 5788824, 5788832, 5788826, 5788926, 5788914, 5788874, 5788828, 5788820, 5788818, 5788830, 5789754, 5789750, 5788876, 5788852, 5789760, 5788840, 5788808, 5788806, 5788842, 5789762, 5788920, 5788900, 5788822, 5788814, 5788810, 5788812, 5788816, 5788858, 5788838, 5788868, 5789854, 5788870, 5788992, 5789966, 5789784, 5789770, 5789004, 5788966, 5789962, 5789860, 5789786, 5789050, 5790024, 5790026, 5789024, 5790232, 5790146, 5789084, 5790148, 5790124, 5790154, 5790118, 5790120, 5790122, 5790116, 5790126, 5790230, 5790228, 5793588, 5790132, 5790128, 5790152, 5790130, 5790134, 5790150, 5789106, 5790144, 5789074, 5790226, 5789026, 5790136, 5789076, 5790140, 5789868, 5789866, 5789864, 1631055, 1631053, 1631525, 1631021, 1631029, 1631001, 1629569, 1629551, 1629811, 1629535, 1631037, 1631613, 1631523, 1630995, 1630989, 1629561, 1630981, 1629517, 1629795, 1629827, 1629509, 1631019, 1629567, 1631017, 1629817, 1629547, 1629813, 1629529, 1631049, 1631041, 1631013, 1629559, 1629819, 1629793, 1631039, 1631615, 1631035, 1631011, 1631005, 1629823, 1631603, 1629829, 1629549, 1631007, 1631605, 1631003, 1629515, 1629799, 1629825, 1629511, 1631689, 1629563, 1630983, 1629545, 1629815, 1629557, 1629831, 1629543, 1630991, 1630993, 1630997, 1629565, 1631015, 1629821, 1629537, 1629533, 1629797, 1629513, 1629807, 1629531, 1629521, 1629527, 1630999, 1631661, 1629553, 1631027, 1631663, 1631057, 1631665, 1629523, 1629525, 1629809, 1629805, 1629803, 1629801]

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def save_dataframe_in_chunks(df, file_path, chunk_size=1000):
    with pd.HDFStore(file_path, mode='w') as store:
        for i, chunk in enumerate(chunks(df.columns, chunk_size)):
            chunk_df = df[chunk]
            key = f'data_{i}'
            store.put(key, chunk_df, format='table')
            print(f"Saved chunk {i} with columns {chunk} to key {key}")

def export_ts_to_hdf(start_date, comids, parameter, rng, mem=None, timesteps=None):
    print(f"Extracting data for {parameter} on {start_date}")
    df = extract_timeseries(
        comids=comids,
        parameter=parameter,
        rng=rng,
        start_date=start_date,
        t0=start_date.hour,
        mem=mem,
        timesteps=timesteps
    )
    
    if isinstance(df, dict):
        for k, v in df.items():
            print(f"Data for key {k}:")
            print(v.head())
    else:
        print(f"Data for parameter {parameter}:")
        print(df.head())

    dir = Path(f"{EXPORT_DIR}/{start_date.strftime('%Y%m%d%H')}")
    dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(df, dict):
        for k, v in df.items():
            file_path = dir / f"{start_date.strftime('%Y%m%d%H')}_{rng}_{parameter}_{k}{f'_{mem}' if mem else ''}.h5"
            print(f"Saving to {file_path}")
            save_dataframe_in_chunks(v, file_path)
    else:
        file_path = dir / f"{start_date.strftime('%Y%m%d%H')}_{rng}_{parameter}{f'_{mem}' if mem else ''}.h5"
        print(f"Saving to {file_path}")
        save_dataframe_in_chunks(df, file_path)

def export_ts_to_hdf_da(start_date, comids, parameter, rng, days):
    print(f"Extracting data assimilation for {parameter} on {start_date}")
    df = extract_timeseries(
        comids=comids,
        parameter=parameter,
        rng=rng,
        start_date=start_date,
        days=days,
        t0=start_date.hour,
    )
    
    if isinstance(df, dict):
        for k, v in df.items():
            print(f"Data for key {k}:")
            print(v.head())
    else:
        print(f"Data for parameter {parameter}:")
        print(df.head())

    dir = Path(f"{EXPORT_DIR}/{start_date.strftime('%Y%m%d%H')}")
    dir.mkdir(parents=True, exist_ok=True)

    if isinstance(df, dict):
        for k, v in df.items():
            file_path = dir / f"{start_date.strftime('%Y%m%d%H')}_data_assimilation_{parameter}_{k}.h5"
            print(f"Saving to {file_path}")
            save_dataframe_in_chunks(v, file_path)
    else:
        file_path = dir / f"{start_date.strftime('%Y%m%d%H')}_data_assimilation_{parameter}.h5"
        print(f"Saving to {file_path}")
        save_dataframe_in_chunks(df, file_path)

def export_ts_to_hdf_no_da(start_date, comids, parameter, rng, days):
    print(f"Extracting no data assimilation for {parameter} on {start_date}")
    df = extract_timeseries(
        comids=comids,
        parameter=parameter,
        rng=rng,
        start_date=start_date,
        days=days,
        t0=start_date.hour,
    )
    
    if isinstance(df, dict):
        for k, v in df.items():
            print(f"Data for key {k}:")
            print(v.head())
    else:
        print(f"Data for parameter {parameter}:")
        print(df.head())

    dir = Path(f"{EXPORT_DIR}/{start_date.strftime('%Y%m%d%H')}")
    dir.mkdir(parents=True, exist_ok=True)

    if isinstance(df, dict):
        for k, v in df.items():
            file_path = dir / f"{start_date.strftime('%Y%m%d%H')}_no_data_assimilation_{k}.h5"
            print(f"Saving to {file_path}")
            save_dataframe_in_chunks(v, file_path)
    else:
        file_path = dir / f"{start_date.strftime('%Y%m%d%H')}_no_data_assimilation_{parameter}.h5"
        print(f"Saving to {file_path}")
        save_dataframe_in_chunks(df, file_path)

def get_last_valid_medium_range(dt: datetime) -> datetime:
    hours_floored = (dt.hour // 6) * 6
    rounded_dt = datetime(dt.year, dt.month, dt.day, hour=hours_floored)
    return rounded_dt

def prepare_urls(date):
    urls = [
        get_nwm_output_url(t0=t, product=prod, timestep=f, date=d, rng="short_range")
        for f in range(1, 19, 1)
        for t in range(0, 24, 1)
        for d in [date]
        for prod in ["channel_rt"]
    ]

    urls2 = [
        get_nwm_output_url(t0=t, product=prod, timestep=0, date=d, rng="analysis_assim")
        for t in range(0, 24, 1)
        for d in [date]
        for prod in ["channel_rt"]
    ]

    urls4 = [
        get_nwm_output_url(t0=t, product=prod, timestep=0, date=d, rng="analysis_assim_no_da")
        for t in range(0, 24, 1)
        for d in [date]
        for prod in ["channel_rt"]
    ]

    urls3 = [
        get_nwm_output_url(
            t0=t,
            product=prod,
            timestep=f,
            date=d,
            rng="medium_range",
            mem=en,
            is_old_medium_range=False,
            is_forcing=False,
        )
        for f in range(1, 205, 1)
        for t in range(0, 24, 6)
        for d in [date]
        for en in [str(x) for x in range(1, 8, 1)]
        for prod in ["channel_rt"]
    ]

    urls5 = [
        get_nwm_output_url(
            t0=t,
            product=prod,
            timestep=f,
            date=d,
            rng="medium_range_blend",
            is_old_medium_range=False,
            is_forcing=False,
        )
        for f in range(1, 241, 1)
        for t in range(0, 24, 6)
        for d in [date]
        for prod in ["channel_rt"]
    ]

    url_list = urls2 + urls4 + urls
    fil_urls = [u for u in url_list if not get_filename(u).exists()]
    tqdm.write(f"{len(url_list)-len(fil_urls)}/{len(url_list)} already exists downloading {len(fil_urls)} files", end="")
    return fil_urls

def download_data(fil_urls, queue=None):
    if not queue:
        pbar = tqdm(total=len(fil_urls), desc="Downloading data", ncols=100)
    try:
        start_time = time.time()
        if len(fil_urls) > 0:
            with multiprocessing.Pool(DL_PROCESSES) as pool:
                f = partial(download_url)
                for _ in pool.imap_unordered(f, fil_urls):
                    if queue:
                        queue.put('download')
                    else:
                        pbar.update(1)
            pool.close()
            pool.join()
        else:
            tqdm.write("All the requested files are cached.", end="")
            if queue:
                queue.put('download_complete')
        if not queue:
            pbar.close()
        tqdm.write(f"downloading took:--- {(time.time() - start_time)/60:.2f} minutes ---",end="")
    except Exception as e:
        if queue:
            queue.put(f'download_error: {e}')
        if not queue:
            pbar.close()
    finally:
        if queue:
            queue.put('download_complete')
        if not queue:
            pbar.close()

def collect_tasks(begin, comid_all):
    tasks = []
    for hour in range(0, 24):
        t0 = begin + DH * hour

        params = ["qBtmVertRunoff", "qSfcLatRunoff", "qBucket", "streamflow"]
        for param in params:
            tasks.append({
                'function': export_ts_to_hdf,
                'args': (t0, comid_all, param, "short_range")
            })

            tasks.append({
                'function': export_ts_to_hdf_da,
                'args': (t0, comid_all, param, "analysis_assim",  1)
            })

            #t0_mrfcst = get_last_valid_medium_range(t0)
            #if t0 == t0_mrfcst:
            #    mem_values = range(1, 8)
            #    for mem in mem_values:
            #        tasks.append({
            #            'function': export_ts_to_hdf,
            #            'args': (t0_mrfcst, comid_all, param, 'medium_range', str(mem), range(1, 205))
            #        })
            #    tasks.append({
            #        'function': export_ts_to_hdf,
            #        'args': (t0, comid_all, param, "medium_range_blend")
            #    })
        params = ["qSfcLatRunoff", "qBucket", "streamflow"]        
        tasks.append({
            'function': export_ts_to_hdf_no_da,
            'args': (t0, comid_all, params, "analysis_assim_no_da", 1)
        })
    return tasks

def extract(tasks, queue):
    start_time_step = time.time()

    def task_composer(tasks, num_processes):
        try:
            #tasks_by_range = {'short_range': [], 'analysis_assim': [], 'medium_range': [], 'analysis_assim_no_da': [], 'medium_range_blend': []}
            tasks_by_range = {'short_range': [], 'analysis_assim': [], 'analysis_assim_no_da': []}
            for task in tasks:
                tasks_by_range[task['args'][3]].append(task)

            for task_range, tasks in tasks_by_range.items():
                for chunk in chunks(tasks, num_processes):
                    processes = []
                    for task in chunk:
                        process = multiprocessing.Process(target=task['function'], args=task['args'])
                        process.start()
                        processes.append(process)

                    for p in processes:
                        queue.put('extract')
                        p.join()
        except Exception as e:
            queue.put(f'extract_error: {e}')
        finally:
            queue.put('extract_complete')

    task_composer(tasks, NUM_PROCESSES)
    tqdm.write(f"Extracting took:--- {(time.time() - start_time_step)/60} minutes ---", end="")

def single_step(begin, comid_all) -> list:
    start_time_step = time.time()
    date_download = copy(begin + DT * 1)
    date_extract = copy(begin)

    url_list = prepare_urls(date_download)
    tasks = collect_tasks(date_extract, comid_all)
    
    pbar1 = tqdm(total=len(url_list), desc=f"Downloading data for {date_download}", ncols=100)
    pbar2 = tqdm(total=len(tasks), desc=f"Extracting data for {date_extract}", ncols=100)

    Q = multiprocessing.Queue()

    p1 = multiprocessing.Process(target=download_data, args=(url_list, Q))
    p2 = multiprocessing.Process(target=extract, args=(tasks, Q))
  
    p1.start()
    p2.start()

    download_complete = False
    extract_complete = False

    while not download_complete or not extract_complete:
        msg = Q.get()
        if msg == 'download':
            pbar1.update(1)
        elif msg == 'extract':
            pbar2.update(1)
        elif msg == 'download_complete':
            download_complete = True
        elif msg == 'extract_complete':
            extract_complete = True
        elif msg.startswith('download_error') or msg.startswith('extract_error'):
            tqdm.write(f"An error occurred: {msg}")
            break

    pbar1.close()
    pbar2.close()
    p1.join()
    p2.join()
    tqdm.write(f"1 step took:--- {(time.time() - start_time_step)/60} minutes ---",end="")

if __name__ == "__main__":
    url_list = prepare_urls(START_DATE)
    download_data(url_list)

    pbar0 = tqdm(total=NUMDAYS, desc=f"Downloading data from {START_DATE} to {START_DATE + NUMDAYS*DT}", ncols=150)
    for day in range(0, NUMDAYS):
        single_step(START_DATE + DT * day, COMIDS)
        pbar0.update(1)
        try:
            shutil.rmtree(f"{CACHE_DIR}/nwm.{(START_DATE + DT * day - DT*CACHED_DAYS).strftime('%Y%m%d')}")
        except Exception as e:
            tqdm.write(f"Removing cached folder failed: {e}")
            continue
    pbar0.close()
