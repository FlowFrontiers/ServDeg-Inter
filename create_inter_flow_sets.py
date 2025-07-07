import gc
import pandas as pd
import json
from pandarallel import pandarallel
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler
import re

OVERLAP_MAX_COUNT = 30


def calculate_overlaps(row, _df, X_cols, application_category_names_list, application_names, OVERLAP_MAX_COUNT):
    import pandas as pd
    idx = row.name

    app_cat_counts = {col: 0 for col in application_category_names_list}
    app_name_counts = {col: 0 for col in application_names}
    no_start = row['O_timestamps'][-1] + row['O_delay'][-1]
    overlap_count = 0
    active_timeout = 1_800_000 # ms

    i = 1
    while idx + i <= _df.index[-1]:
        next_flow = _df.loc[idx+i]
        if next_flow['day'] != row['day']:
            break
        if next_flow['location'] == row['location'] and next_flow['O_timestamps'][0] >= no_start:
            break
        i += 1

    overlaps = []
    while idx + i <= _df.index[-1] \
            and next_flow['day'] == row['day'] \
            and overlap_count < OVERLAP_MAX_COUNT \
            and next_flow['O_timestamps'][-1] + next_flow['O_delay'][-1] <= row['O_timestamps'][0] + active_timeout:
        next_flow = _df.loc[idx+i]
        if next_flow['location'] == row['location']:
            relative_start = next_flow['O_timestamps'][0] - no_start
            app_cat_counts[f"covering_count_appcat_{next_flow['application_category_name']}"] += 1
            app_name_counts[f"covering_count_app_{next_flow['application_name']}"] += 1
            overlaps.append([relative_start] + next_flow[X_cols].to_list())
            overlap_count += 1
        i += 1

    overlaps += [ [None] * (len(X_cols) + 1) ] * (OVERLAP_MAX_COUNT - len(overlaps))
    overlaps = [item for flow_data in overlaps for item in flow_data]
    overlaps += list(app_cat_counts.values())
    overlaps += list(app_name_counts.values())
    return pd.Series(overlaps)


if __name__ == '__main__':
    pandarallel.initialize(progress_bar=True)

    days = ['MON', 'TUE', 'WED', 'THU', 'FRI']
    Ms = [5, 10, 15, 20]
    SAMPLE_THRESHOLD = 100_000
    tqdm.pandas()

    with open('setup.json', 'r') as openfile:
        setup_object = json.load(openfile)
        WD = setup_object["wd_path"]

    for M in Ms:
        path = os.path.join(WD, "train_test", "inter", f"M{M}")
        try:
            os.makedirs(path)
        except OSError as err:
            pass

        print(f'Processing M={M}')
        dfs = []
        for idx, day in enumerate(days):
            df = pd.read_parquet(f"{WD}/preprocessed/v4/M{M}/{day}v4.parquet",
                                 columns=['application_category_name', 'application_name',
                                          'O_delay', 'O_timestamps', 'location'])
            df = df[df['O_delay'].apply(len) == M]
            df['day'] = idx
            dfs.append(df)

        v4_dfs = pd.concat(dfs).reset_index(drop=True)
        application_category_name_cols = [f"covering_count_appcat_{col}" for col in
                                          v4_dfs['application_category_name'].unique()]
        application_name_cols = [f"covering_count_app_{col}" for col in v4_dfs['application_name'].unique()]

        for stage, dfs_preprocessed in zip(['train', 'test'], [dfs[:3], dfs[3:]]):
            _X = pd.read_parquet(f'{WD}/train_test/intra/M{M}/X_{stage}.parquet')
            y_regression_count = pd.read_parquet(f'{WD}/train_test/intra/M{M}/y_{stage}_regression_count.parquet')
            y_regression_max_len = pd.read_parquet(
                f'{WD}/train_test/intra/M{M}/y_{stage}_regression_max_len.parquet')
            y_regression_max_start = pd.read_parquet(
                f'{WD}/train_test/intra/M{M}/y_{stage}_regression_max_start.parquet')
            y_regression_max_end = pd.read_parquet(
                f'{WD}/train_test/intra/M{M}/y_{stage}_regression_max_end.parquet')
            v4_dfs = pd.concat(dfs_preprocessed)
            v4_dfs.index = _X.index
            Xy = pd.concat([v4_dfs, _X, y_regression_count, y_regression_max_len, y_regression_max_start,
                                               y_regression_max_end],
                           axis=1)
            Xy['O_start'] = Xy['O_timestamps'].apply(lambda x: x[0])
            Xy = Xy.sort_values(by=['day', 'O_start'], ascending=[True, True]).reset_index(drop=True)
            Xy_sampled = Xy if len(Xy) <= SAMPLE_THRESHOLD else Xy.sample(n=SAMPLE_THRESHOLD,
                                                                          random_state=42)

            y_classifier = Xy_sampled['NO_SD_count'].apply(lambda x: x > 0)
            y_regression_count = Xy_sampled['NO_SD_count']
            y_regression_max_len = Xy_sampled['NO_SD_max_len']
            y_regression_max_start = Xy_sampled['NO_SD_max_start']
            y_regression_max_end = Xy_sampled['NO_SD_max_end']

            X = Xy.drop(columns=['NO_SD_count', 'NO_SD_max_len', 'NO_SD_max_start', 'NO_SD_max_end'])
            X_sampled = Xy_sampled.drop(columns=['NO_SD_count', 'NO_SD_max_len', 'NO_SD_max_start', 'NO_SD_max_end'])

            X_cols = [col for col in _X.columns.to_list() if not col.startswith('application_category_name_') and
                      not col.startswith('application_name_') and
                      not col.startswith('location_') and
                      col != 'connection_type_wireless']

            new_X_cols = []
            for i in range(OVERLAP_MAX_COUNT):
                new_X_cols += [f'ov_{i + 1}_{col}' for col in ['relative_start_ms'] + X_cols]
            new_X_cols += application_category_name_cols
            new_X_cols += application_name_cols

            X_new_cols = X_sampled.parallel_apply(calculate_overlaps,
                                                  _df=X,
                                                  X_cols=X_cols,
                                                  application_category_names_list=application_category_name_cols,
                                                  application_names=application_name_cols,
                                                  OVERLAP_MAX_COUNT=OVERLAP_MAX_COUNT,
                                                  axis=1)
            X_new_cols.columns = new_X_cols
            X_sampled = X_sampled.drop(columns=['application_category_name',
                                                'application_name',
                                                'O_start',
                                                'O_delay',
                                                'O_timestamps',
                                                'location',
                                                'day'])

            path_temp = os.path.join(WD, "train_test", "inter", f"M{M}", "temp")
            try:
                os.makedirs(path_temp)
            except OSError as err:
                pass

            X_sampled.to_parquet(f'{path_temp}/X_{stage}.parquet')
            X_new_cols.to_parquet(f'{path_temp}/X_new_cols_{stage}.parquet')
            pd.DataFrame(y_classifier).to_parquet(f'{path_temp}/y_{stage}_classifier.parquet')
            pd.DataFrame(y_regression_count).to_parquet(f'{path_temp}/y_{stage}_regression_count.parquet')
            pd.DataFrame(y_regression_max_len).to_parquet(f'{path_temp}/y_{stage}_regression_max_len.parquet')
            pd.DataFrame(y_regression_max_start).to_parquet(f'{path_temp}/y_{stage}_regression_max_start.parquet')
            pd.DataFrame(y_regression_max_end).to_parquet(f'{path_temp}/y_{stage}_regression_max_end.parquet')

            Xy = pd.concat([X_sampled, X_new_cols], axis=1)
            Xy['y_classifier'] = y_classifier
            Xy['y_regression_count'] = y_regression_count
            Xy['y_regression_max_len'] = y_regression_max_len
            Xy['y_regression_max_start'] = y_regression_max_start
            Xy['y_regression_max_end'] = y_regression_max_end
            Xy.to_parquet(f'{path}/Xy_{stage}.parquet')

            del X
            del X_sampled
            del X_new_cols
            del Xy
            del Xy_sampled
            gc.collect()



    for M in tqdm(Ms):
        path = os.path.join(WD, "train_test", "inter", f"M{M}")
        X_merged = []
        train_amount = 0

        for stage in ['train', 'test']:
            X = pd.read_parquet(f'{path}/temp/X_{stage}.parquet')
            pattern = re.compile(
                r"ov_\d+_(O_(start|delay|timestamps)|application_name|application_category_name|day|location)")
            drop_columns = [col for col in X.columns.to_list() if pattern.match(col) or 'connection_type_wired' in col]
            X.drop(columns=drop_columns, inplace=True)

            X_cols = [col for col in X.columns.to_list() if 'application_category_name_' not in col and
                      'application_name_' not in col and
                      'location_' not in col and
                      col != 'connection_type_wireless']

            X_new_cols = pd.read_parquet(f'{path}/temp/X_new_cols_{stage}.parquet')
            drop_columns = [col for col in X_new_cols.columns.to_list() if pattern.match(col) or 'connection_type_wired' in col]
            X_new_cols.drop(columns=drop_columns, inplace=True)

            X_merged.append(pd.concat([X, X_new_cols], axis=1))
            if stage == 'train':
                train_amount = len(X)

        X_merged = pd.concat(X_merged)

        # Data Scaling
        X_scaled = X_merged.copy()
        features_to_scale = X_scaled[X_cols]
        scaler = StandardScaler().fit(features_to_scale.values)
        features_scaled = scaler.transform(features_to_scale.values)
        X_scaled[X_cols] = features_scaled

        # Simple Feature Selection: drop the columns with no variation
        no_variation_columns = X_merged.columns[X_merged.nunique() <= 1]
        X_merged.drop(columns=no_variation_columns, inplace=True)
        X_scaled.drop(columns=no_variation_columns, inplace=True)

        X_merged.iloc[:train_amount].to_parquet(f'{path}/X_train.parquet')
        X_scaled.iloc[:train_amount].to_parquet(f'{path}/X_train_scaled.parquet')
        X_merged.iloc[train_amount:].to_parquet(f'{path}/X_test.parquet')
        X_scaled.iloc[train_amount:].to_parquet(f'{path}/X_test_scaled.parquet')

        del X
        del X_merged
        del X_new_cols
        del X_scaled
        gc.collect()