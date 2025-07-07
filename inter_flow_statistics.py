import pandas as pd
from tqdm import tqdm
import json
import os
import glob
import gc


def get_overlapping_observable_part(row, start, end):
    if start <= row['O_timestamps'][-1] and end >= row['O_timestamps'][0] + row['O_delay'][0]:
        try:
            start_index = next(x for x, val in enumerate(row['O_timestamps']) if val >= start)
        except StopIteration:
            start_index = len(row['O_timestamps']) - 1

        O_end_timestamps = [ts + d for ts, d in zip(row['O_timestamps'], row['O_delay'])]
        try:
            end_index = next(x for x, val in enumerate(O_end_timestamps) if val >= end)
        except StopIteration:
            end_index = len(row['O_timestamps'])

        start_timestamp = row['O_timestamps'][start_index]
        start_timestamp = start_timestamp if start_timestamp > start else start
        end_timestamp = row['O_timestamps'][end_index - 1] + row['O_delay'][end_index - 1]
        end_timestamp = end_timestamp if end_timestamp < end else end

        SDs = []
        for SD_s_ts, SD_e_ts in row['O_SD_sequences_timestamps']:
            if SD_s_ts < end_timestamp and SD_e_ts > start_timestamp:
                SDs.append((max(SD_s_ts, start_timestamp), min(SD_e_ts, end_timestamp)))
        return row['O_delay'][start_index: end_index], start_timestamp, end_timestamp, SDs
    return None

def build_statistics(row, overlap_list):
    # overlap_list = row['NO_overlaps']
    count_overlaps = len(overlap_list)
    count_overlapping_SDs = 0

    overlapping_rel_idxs = []
    overlapping_rel_idxs_with_SD = []
    overlapping_rel_idxs_same_appcat = []
    overlapping_rel_idxs_same_app = []
    overlapping_rel_idxs_same_appcat_with_SD = []
    overlapping_rel_idxs_same_app_with_SD = []

    overlapping_rel_time = []
    overlapping_rel_time_with_SD = []
    overlapping_rel_time_same_appcat = []
    overlapping_rel_time_same_app = []
    overlapping_rel_time_same_appcat_with_SD = []
    overlapping_rel_time_same_app_with_SD = []

    SDs_covering_portions = []
    SDs_center_distances = []

    # for idx, _, app_cat, app, _, end_time, _, SDs in overlap_list:
    for idx, app_cat, app, _, end_time, _, SDs in overlap_list:
        overlapping_rel_idxs.append(idx - row.name)
        overlapping_rel_time.append(end_time - row['NO_timestamps'][0])
        if app_cat == row['application_category_name']:
            overlapping_rel_idxs_same_appcat.append(idx - row.name)
            overlapping_rel_time_same_appcat.append(end_time - row['NO_timestamps'][0])
            if len(SDs) > 0:
                overlapping_rel_idxs_same_appcat_with_SD.append(idx - row.name)
                overlapping_rel_time_same_appcat_with_SD.append(end_time - row['NO_timestamps'][0])
        if app == row['application_name']:
            overlapping_rel_idxs_same_app.append(idx - row.name)
            overlapping_rel_time_same_app.append(end_time - row['NO_timestamps'][0])
            if len(SDs) > 0:
                overlapping_rel_idxs_same_app_with_SD.append(idx - row.name)
                overlapping_rel_time_same_app_with_SD.append(end_time - row['NO_timestamps'][0])
        if len(SDs) > 0:
            overlapping_rel_idxs_with_SD.append(idx - row.name)
            overlapping_rel_time_with_SD.append(end_time - row['NO_timestamps'][0])

        count_overlapping_SDs += len(SDs)
        for SD_overlap_s, SD_overlap_e in SDs:
            portions = []
            center_distances = []
            for SD_original_s, SD_original_e in row['NO_SD_sequences_timestamps']:
                portions.append((SD_overlap_e - SD_overlap_s) / (SD_original_e - SD_original_s))
                center_distances.append(SD_overlap_s - SD_original_s + (SD_overlap_e - SD_overlap_s - SD_original_e + SD_original_s) / 2)
            SDs_covering_portions.append(max(portions))
            SDs_center_distances.append(min(center_distances))


    count_overlaps_with_SD = len(overlapping_rel_idxs_with_SD)
    count_overlaps_same_appcat = len(overlapping_rel_idxs_same_appcat)
    count_overlaps_same_app = len(overlapping_rel_idxs_same_app)
    count_overlaps_same_appcat_with_SD = len(overlapping_rel_idxs_same_appcat_with_SD)
    count_overlaps_same_app_with_SD = len(overlapping_rel_idxs_same_app_with_SD)

    return [count_overlaps, count_overlaps_with_SD, count_overlapping_SDs, overlapping_rel_idxs, overlapping_rel_idxs_with_SD,
            count_overlaps_same_appcat, overlapping_rel_idxs_same_appcat,
            count_overlaps_same_app, overlapping_rel_idxs_same_app,
            count_overlaps_same_appcat_with_SD, overlapping_rel_idxs_same_appcat_with_SD,
            count_overlaps_same_app_with_SD, overlapping_rel_idxs_same_app_with_SD,
            overlapping_rel_time, overlapping_rel_time_with_SD, overlapping_rel_time_same_appcat,
            overlapping_rel_time_same_app, overlapping_rel_time_same_appcat_with_SD,
            overlapping_rel_time_same_app_with_SD,
            SDs_covering_portions, SDs_center_distances]

def calculate_relative_to_relative_indexes(overlapping_rel_idxs,
                                           overlapping_rel_idxs_with_SD,
                                           overlapping_rel_idxs_same_appcat,
                                           overlapping_rel_idxs_same_app,
                                           overlapping_rel_idxs_same_appcat_with_SD,
                                           overlapping_rel_idxs_same_app_with_SD
                                           ):
    if overlapping_rel_idxs == []:
        return [[], [], [], [], [], []]

    rel_base_index = overlapping_rel_idxs[0]
    rel_idxs = [x - rel_base_index for x in overlapping_rel_idxs]
    rel_idxs_with_SD = [x - rel_base_index for x in overlapping_rel_idxs_with_SD]
    rel_idxs_with_same_appcat = [x - rel_base_index for x in overlapping_rel_idxs_same_appcat]
    rel_idxs_with_same_app = [x - rel_base_index for x in overlapping_rel_idxs_same_app]
    rel_idxs_with_same_appcat_with_SD = [x - rel_base_index for x in overlapping_rel_idxs_same_appcat_with_SD]
    rel_idxs_with_same_app_with_SD = [x - rel_base_index for x in overlapping_rel_idxs_same_app_with_SD]
    return [rel_idxs, rel_idxs_with_SD,
            rel_idxs_with_same_appcat, rel_idxs_with_same_appcat_with_SD,
            rel_idxs_with_same_app, rel_idxs_with_same_app_with_SD]


''' Check all flows that have overlaps (no matter how small they may be) '''
def check_through_overlaps_all(row, _df):
    idx = row.name
    overlaps = []
    if len(row['NO_SD_sequences_idx']) > 0:
        no_start = row['NO_timestamps'][0]
        no_end = row['NO_timestamps'][-1]

        i = 1
        while idx + i < len(_df):
            next_flow = _df.loc[idx + i]
            overlapping_part = get_overlapping_observable_part(next_flow, no_start, no_end)
            if overlapping_part != None:
                delays, start_timestamp, end_timestamp, SDs = overlapping_part
                overlaps.append((idx + i,
                                 next_flow['application_category_name'],
                                 next_flow['application_name'],
                                 start_timestamp,
                                 end_timestamp,
                                 delays,
                                 SDs
                                 ))
            i += 1
    stats = build_statistics(row, overlaps)
    overlapping_rel_idxs = stats[3]
    overlapping_rel_idxs_with_SD = stats[4]
    overlapping_rel_idxs_same_appcat = stats[6]
    overlapping_rel_idxs_same_app = stats[8]
    overlapping_rel_idxs_same_appcat_with_SD = stats[10]
    overlapping_rel_idxs_same_app_with_SD = stats[12]
    rel_idxs = calculate_relative_to_relative_indexes(overlapping_rel_idxs,
                                                      overlapping_rel_idxs_with_SD,
                                                      overlapping_rel_idxs_same_appcat,
                                                      overlapping_rel_idxs_same_app,
                                                      overlapping_rel_idxs_same_appcat_with_SD,
                                                      overlapping_rel_idxs_same_app_with_SD)
    return pd.Series(stats + rel_idxs)


''' Check only the flows that will be used in the model evaluation '''
def check_through_overlaps_used(row, _df):
    idx = row.name
    overlaps = []
    active_timeout = 1_800_000 # ms
    
    if len(row['NO_SD_sequences_idx']) > 0:
        no_start = row['NO_timestamps'][0]
        no_end = row['O_timestamps'][0] + active_timeout  # theoretical max end instead of real end

        i = 1
        while idx + i <= _df.index[-1]:
            next_flow = _df.loc[idx + i]
            if next_flow['O_timestamps'][0] >= no_start:
                break
            i += 1
        
        while idx + i <= _df.index[-1] \
        and next_flow['O_timestamps'][-1] + next_flow['O_delay'][-1] <= row['O_timestamps'][0] + active_timeout:
            next_flow = _df.loc[idx + i]
            overlapping_part = get_overlapping_observable_part(next_flow, no_start, no_end)
            if overlapping_part != None:
                delays, start_timestamp, end_timestamp, SDs = overlapping_part
                overlaps.append((idx + i,
                                 next_flow['application_category_name'],
                                 next_flow['application_name'],
                                 start_timestamp,
                                 end_timestamp,
                                 delays,
                                 SDs
                                 ))
            i += 1
    stats = build_statistics(row, overlaps)
    overlapping_rel_idxs = stats[3]
    overlapping_rel_idxs_with_SD = stats[4]
    overlapping_rel_idxs_same_appcat = stats[6]
    overlapping_rel_idxs_same_app = stats[8]
    overlapping_rel_idxs_same_appcat_with_SD = stats[10]
    overlapping_rel_idxs_same_app_with_SD = stats[12]
    rel_idxs = calculate_relative_to_relative_indexes(overlapping_rel_idxs,
                                                      overlapping_rel_idxs_with_SD,
                                                      overlapping_rel_idxs_same_appcat,
                                                      overlapping_rel_idxs_same_app,
                                                      overlapping_rel_idxs_same_appcat_with_SD,
                                                      overlapping_rel_idxs_same_app_with_SD)
    return pd.Series(stats + rel_idxs)



def create_statistics(WD, days, M, stats_for_training):
    directory = "inter_stats_used" if stats_for_training else "inter_stats_all"
    for day in days:
        df = pd.read_parquet(f"{WD}/preprocessed/v4/M{M}/{day}v4.parquet")
        if stats_for_training:
            # Reduce the examined flows to the ones that have full O parts
            df = df[df['O_delay'].apply(len) == M].reset_index(drop=True)
            df['O_start'] = df['O_timestamps'].apply(lambda x: x[0])
            df = df.sort_values(by='O_start') # Sort by the start of the O part

        unique_locations = df['location'].unique()
        for l_idx, location in enumerate(unique_locations):
            print(f'{day} L{l_idx}/{len(unique_locations)} ({location})')
            df_location_split = df[df['location'] == location].reset_index()

            df_location_split[['count_overlaps', 'count_overlaps_with_SD', 'count_overlapping_SDs', 'overlapping_rel_idxs',
                               'overlapping_rel_idxs_with_SD',
                               'count_overlaps_same_appcat', 'overlapping_rel_idxs_same_appcat',
                               'count_overlaps_same_app', 'overlapping_rel_idxs_same_app',
                               'count_overlaps_same_appcat_with_SD', 'overlapping_rel_idxs_same_appcat_with_SD',
                               'count_overlaps_same_app_with_SD', 'overlapping_rel_idxs_same_app_with_SD',
                               'overlapping_rel_time', 'overlapping_rel_time_with_SD', 'overlapping_rel_time_same_appcat',
                               'overlapping_rel_time_same_app', 'overlapping_rel_time_same_appcat_with_SD',
                               'overlapping_rel_time_same_app_with_SD', 'SDs_covering_portions', 'SDs_center_distances',
                               'overlapping_rel_to_rel_idxs', 'overlapping_rel_to_rel_idxs_with_SD',
                               'overlapping_rel_to_rel_idxs_same_appcat', 'overlapping_rel_to_rel_idxs_same_appcat_with_SD',
                               'overlapping_rel_to_rel_idxs_same_app',
                               'overlapping_rel_to_rel_idxs_same_app_with_SD']] = df_location_split.progress_apply(check_through_overlaps_used,
                                                                                                           _df=df_location_split,
                                                                                                           axis=1)

            path = os.path.join(WD, directory, f'M{M}_split_res')
            try:
                os.makedirs(path)
            except OSError as err:
                pass
            df_location_split.to_parquet(f'{path}/{day}_L{location}_v5.parquet')

            del df_location_split


if __name__ == "__main__":
    days = ['MON', 'TUE', 'WED']
    Ms = [5, 10]
    stats_for_training = True
    directory = "inter_stats_used" if stats_for_training else "inter_stats_all"
        
    tqdm.pandas()

    with open('setup.json', 'r') as openfile:
        setup_object = json.load(openfile)
        WD = setup_object["wd_path"]

    # Create the statistics
    for M in Ms:
        print(f'M={M}')
        create_statistics(WD, days, M, stats_for_training)

    for M in Ms:
        path = os.path.join(WD, directory, f'M{M}_split_res')
        all_files = glob.glob(os.path.join(WD, directory, f'M{M}_split_res', "*.parquet"))

        dfs = []
        for filename in all_files:
            dfs.append(pd.read_parquet(filename))
        df = pd.concat(dfs)
        del dfs
        gc.collect()
        if len(df) > 500_000:
            df.sample(n=500_000).to_parquet(f'{WD}/{directory}/M{M}.parquet')
        else:
            df.to_parquet(f'{WD}/{directory}/M{M}.parquet')
        del df
        gc.collect()
