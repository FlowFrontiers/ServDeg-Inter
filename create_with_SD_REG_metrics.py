import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, median_absolute_error, r2_score
import json


def reverse_onehot(row, prefix='application_category_name_'):
    for col in row.index:
        if row[col] == 1:
            return col.replace(prefix, '')
    return None


if __name__ == "__main__":
    with open('setup.json', 'r') as openfile:
        setup_object = json.load(openfile)
        WD = setup_object["wd_path"]

    with open(f"{WD}/min_seq_lens.json", 'r') as sequencefile:
        min_seq_lens = json.load(sequencefile)

    for y_test_name, preds_name in zip(['y_test_regression_count', 'y_test_regression_max_len',
                                           'y_test_regression_max_start', 'y_test_regression_max_end'],
                                          ['REG_count_predictions', 'REG_max_len_predictions',
                                           'REG_max_start_predictions', 'REG_max_end_predictions']
                                           ):
        for M in [5, 10]:
            y_test_classifier = pd.read_parquet(f'{WD}/train_test/inter/M{M}/y_test_classifier.parquet')
            X_test = pd.read_parquet(f'{WD}/train_test/inter/M{M}/X_test.parquet')
            X_test['application_category_name'] = X_test.filter(regex='^application_category_name').apply(
                reverse_onehot, axis=1)
            # X_test['M'] = M
            y_test = pd.read_parquet(f'{WD}/train_test/inter/M{M}/{y_test_name}.parquet').iloc[:, 0]
            y_preds = pd.read_parquet(f'{WD}/inter_results/M{M}/{preds_name}.parquet')

            df = pd.concat([X_test, y_test_classifier], axis=1)
            df['y'] = y_test
            df = df.reset_index()

            df = pd.concat([df, y_preds], axis=1)
            df = df[df['NO_SD_count'] > 0]

            df_metrics = pd.DataFrame(columns=['Model', 'MAE', 'RMSE', 'MAPE', 'MedianAE', '$R^2$'])
            df_metrics = df_metrics.set_index('Model')
            for model_col in ['Ridge Regression', 'XGBoost', 'MLP']:
                mae = mean_absolute_error(df['y'], df[model_col])
                rmse = root_mean_squared_error(df['y'], df[model_col])
                mape = mean_absolute_percentage_error(df['y'], df[model_col])
                medae = median_absolute_error(df['y'], df[model_col])
                r2 = r2_score(df['y'], df[model_col])

                df_metrics.loc[model_col] = [mae, rmse, mape, medae, r2]
                df_metrics.to_csv(f"{WD}/inter_results/M{M}/{preds_name.replace('predictions', 'metrics')}_with_SD.csv")
