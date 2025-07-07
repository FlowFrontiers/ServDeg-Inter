import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
import math

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import random
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, median_absolute_error, r2_score


def fit_test_classifiers(X_train, X_train_scaled, y_train_classifier, X_test, X_test_scaled, y_test_classifier, save_path):
    # Logistic Regression parameters
    log_reg_params = {
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10],
        'max_iter': [1000]
    }

    # XGBoost parameters
    xgb_params = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    
    # MLP parameters
    mlp_params = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }


    classification_model_names = ['Logistic Regression', 'XGBoost', 'MLP']


    # ------------ GridSearchCV with Stratified-5-fold cross-validation ------------
    scoring = 'roc_auc'
    # Logistic Regression
    log_reg_grid = GridSearchCV(LogisticRegression(random_state=42), log_reg_params, cv=5, scoring=scoring, n_jobs=-1)
    log_reg_grid.fit(X_train, y_train_classifier)
    print("LOG_REG fitted")
    # XGBoost
    xgb_grid = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), xgb_params,
                            cv=5, scoring=scoring, n_jobs=-1)
    xgb_grid.fit(X_train, y_train_classifier)
    print("XGB fitted")
    # MLP
    mlp_grid = GridSearchCV(estimator=MLPClassifier(max_iter=1000), param_grid=mlp_params, cv=5, n_jobs=-1)
    mlp_grid.fit(X_train_scaled, y_train_classifier)
    print("MLP fitted")

    # ------------ Extract best params ------------
    classification_grids = [log_reg_grid, xgb_grid, mlp_grid]
    classification_best_params = pd.DataFrame()
    classification_best_params['model'] = classification_model_names
    classification_best_params['parameters'] = [grid.best_params_ for grid in classification_grids]

    # ------------ Make predictions for all trained models  ------------
    classification_models = [grid.best_estimator_ for grid in classification_grids]
    # :-1 To exclude MLP which is added separately as predicting on the scaled data
    classification_predictions = np.array(
        [model.predict(X_test) for model in classification_models[:-1]]
        + [ classification_models[-1].predict(X_test_scaled) ]
    )
    classification_probs = np.array(
        [model.predict_proba(X_test)[:, 1] for model in classification_models[:-1]]
        + [ classification_models[-1].predict_proba(X_test_scaled)[:, 1] ]
    )


    # ------------ Evaluate prediction metrics  ------------
    classification_metrics_data = []
    for model, preds in zip(classification_model_names, classification_predictions):
        accuracy = accuracy_score(y_test_classifier, preds)
        precision = precision_score(y_test_classifier, preds)
        recall = recall_score(y_test_classifier, preds)
        f1 = f1_score(y_test_classifier, preds)
        tn, fp, fn, tp = confusion_matrix(y_test_classifier, preds).ravel()
        specificity = tn / (tn + fp)  # True Negative Rate
        npv = tn / (tn + fn)  # Negative Predictive Value (~precision for negative class)
        balanced_accuracy = (recall + specificity) / 2

        classification_metrics_data.append({
            'Model': model,
            'Precision': precision,
            'Recall (TPR)': recall,
            '$F_1$-score': f1,
            'Specificity (TNR)': specificity,
            'NPV': npv,
            'Accuracy': accuracy,
            'Balanced Accuracy': balanced_accuracy
        })

    # Create DataFrame
    classification_metrics_df = pd.DataFrame(classification_metrics_data)
    # Set the index to 'Model' for easier plotting
    classification_metrics_df.set_index('Model', inplace=True)

    # Save the results
    pd.DataFrame(classification_predictions.transpose(), columns=classification_model_names).to_parquet(f'{save_path}/CLASS_predictions.parquet')
    pd.DataFrame(classification_probs.transpose(), columns=classification_model_names).to_parquet(f'{save_path}/CLASS_probs.parquet')
    classification_best_params.to_csv(f'{save_path}/CLASS_best_params.csv')
    classification_metrics_df.to_csv(f'{save_path}/CLASS_metrics.csv')



def fit_test_regressors(X_train, X_train_scaled, y_train_regression, X_test, X_test_scaled, y_test_regression,
                        save_path, type):
    # Ridge Regression parameters
    ridge_params = {
        'alpha': [0.1, 1, 10]
    }


    # XGBoost parameters
    xgb_params = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    # MLP parameters
    mlp_params = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

    regression_model_names = ['Ridge Regression', 'XGBoost', 'MLP']


    # ------------ GridSearchCV with Stratified-5-fold cross-validation ------------
    # Ridge Regression
    ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    ridge_grid.fit(X_train, y_train_regression)
    print("Ridge Regression fitted")
    # XGBoost
    xgb_grid = GridSearchCV(XGBRegressor(), xgb_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    xgb_grid.fit(X_train, y_train_regression)
    print("XGB fitted")
    # MLP
    mlp_grid = GridSearchCV(estimator=MLPRegressor(max_iter=1000), param_grid=mlp_params, cv=5, n_jobs=-1)
    mlp_grid.fit(X_train_scaled, y_train_regression)
    print("MLP fitted")

    regression_grids = [ridge_grid, xgb_grid, mlp_grid]


    # ------------ Extract best params ------------
    regression_best_params = pd.DataFrame()
    regression_best_params['model'] = regression_model_names
    regression_best_params['parameters'] = [grid.best_params_ for grid in regression_grids]


    # ------------ Make predictions for all trained models  ------------
    regression_models = [grid.best_estimator_ for grid in regression_grids]
    regression_predictions = np.array(
        [model.predict(X_test) for model in regression_models[:-1]] +
        [regression_models[-1].predict(X_test_scaled)])


    # ------------ Evaluate prediction metrics  ------------
    regression_metrics_data = []
    for model, preds in zip(regression_model_names, regression_predictions):
        mae = mean_absolute_error(y_test_regression, preds)
        rmse = math.sqrt(mean_squared_error(y_test_regression, preds))
        mape = mean_absolute_percentage_error(y_test_regression, preds)
        medae = median_absolute_error(y_test_regression, preds)
        r2 = r2_score(y_test_regression, preds)

        regression_metrics_data.append({
            'Model': model,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'MedianAE': medae,
            '$R^2$': r2,
        })

    # Create DataFrame
    regression_metrics_df = pd.DataFrame(regression_metrics_data)
    # Set the index to 'Model' for easier plotting
    regression_metrics_df.set_index('Model', inplace=True)
    print(regression_metrics_df)
    #
    # Save the results
    pd.DataFrame(regression_predictions.transpose(), columns=regression_model_names).to_parquet(
        f'{save_path}/REG_{type}_predictions.parquet')
    regression_best_params.to_csv(f'{save_path}/REG_{type}_best_params.csv')
    regression_metrics_df.to_csv(f'{save_path}/REG_{type}_metrics.csv')





if __name__ == "__main__":
    # Ms = [5, 10, 15, 20]
    Ms = [15, 20]

    with open('setup.json', 'r') as openfile:
        setup_object = json.load(openfile)
        WD = setup_object["wd_path"]

    with open(f"{WD}/min_seq_lens.json", 'r') as sequencefile:
        min_seq_lens = json.load(sequencefile)

    for M in tqdm(Ms):
        path = os.path.join(WD, "inter_results", f"M{M}")
        try:
            os.makedirs(path)
        except OSError as err:
            pass

        X_train = pd.read_parquet(f'{WD}/train_test/inter/M{M}/X_train.parquet').fillna(-1)
        X_test = pd.read_parquet(f'{WD}/train_test/inter/M{M}/X_test.parquet').fillna(-1)
        X_train_scaled = pd.read_parquet(f'{WD}/train_test/inter/M{M}/X_train_scaled.parquet').fillna(-1)
        X_test_scaled = pd.read_parquet(f'{WD}/train_test/inter/M{M}/X_test_scaled.parquet').fillna(-1)
        y_train_classifier = pd.read_parquet(f'{WD}/train_test/inter/M{M}/y_train_classifier.parquet')['NO_SD_count']
        y_test_classifier = pd.read_parquet(f'{WD}/train_test/inter/M{M}/y_test_classifier.parquet')['NO_SD_count']
        y_train_regression_count = pd.read_parquet(f'{WD}/train_test/inter/M{M}/y_train_regression_count.parquet')['NO_SD_count']
        y_test_regression_count = pd.read_parquet(f'{WD}/train_test/inter/M{M}/y_test_regression_count.parquet')['NO_SD_count']
        y_train_regression_max_len = pd.read_parquet(f'{WD}/train_test/inter/M{M}/y_train_regression_max_len.parquet')['NO_SD_max_len']
        y_test_regression_max_len = pd.read_parquet(f'{WD}/train_test/inter/M{M}/y_test_regression_max_len.parquet')['NO_SD_max_len']
        y_train_regression_max_start = pd.read_parquet(f'{WD}/train_test/inter/M{M}/y_train_regression_max_start.parquet')['NO_SD_max_start']
        y_test_regression_max_start = pd.read_parquet(f'{WD}/train_test/inter/M{M}/y_test_regression_max_start.parquet')['NO_SD_max_start']
        y_train_regression_max_end = pd.read_parquet(f'{WD}/train_test/inter/M{M}/y_train_regression_max_end.parquet')['NO_SD_max_end']
        y_test_regression_max_end = pd.read_parquet(f'{WD}/train_test/inter/M{M}/y_test_regression_max_end.parquet')['NO_SD_max_end']

        fit_test_classifiers(X_train, X_train_scaled, y_train_classifier, X_test, X_test_scaled, y_test_classifier, path)
        fit_test_regressors(X_train, X_train_scaled, y_train_regression_count, X_test, X_test_scaled, y_test_regression_count, path, 'count')
        fit_test_regressors(X_train, X_train_scaled, y_train_regression_max_len, X_test, X_test_scaled, y_test_regression_max_len, path, 'max_len')
        fit_test_regressors(X_train, X_train_scaled, y_train_regression_max_start, X_test, X_test_scaled, y_test_regression_max_start, path, 'max_start')
        fit_test_regressors(X_train, X_train_scaled, y_train_regression_max_end, X_test, X_test_scaled, y_test_regression_max_end, path, 'max_end')