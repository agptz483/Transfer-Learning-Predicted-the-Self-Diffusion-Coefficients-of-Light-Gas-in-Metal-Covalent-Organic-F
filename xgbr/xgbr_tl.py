import numpy as np
import pandas as pd
import optuna
from xgboost import XGBRegressor as xgbr
from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

def train_trada_xgbr(N, steps, fold, random_state, sample_size, x_train, y_train, target_features, target_target):
    global R2_aoc,save_
    model = TwoStageTrAdaBoostR2(xgbr(n_estimators=187,
                                      learning_rate=0.08,
                                      max_depth=8),
                                  n_estimators=N, sample_size=sample_size,
                                  steps=steps, fold=fold,
                                  random_state=random_state)


    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    y_pred = model.predict(target_features)
    target_mse = mean_squared_error(target_target, y_pred)
    target_r2 = r2_score(target_target, y_pred)

    print('xgbr')
    print('Train R2:', round(train_r2, 5), '\nTrain MSE:', train_mse)
    print('Test R2:', round(target_r2, 5), '\nTest MSE:', target_mse)
    if target_r2 > R2_aoc and save_:
        print('R2_aoc', R2_aoc)
        joblib.dump(model, 'best_tradaboost_xgbr_model.pkl')
        R2_aoc = target_r2
    return target_r2,y_pred


def objective_xgbr(trial):
    n_estimators = trial.suggest_int('n_estimators', 5, 20)
    steps = trial.suggest_int('steps', 2, 10)
    random_state = np.random.RandomState(1)

    R2,_ = train_trada_xgbr(n_estimators, steps, 5, random_state, sample_size, x_train, y_train, target_features,
                        target_target)
    return R2


if __name__ == '__main__':

    source_data = pd.read_csv('F:/work_github/mof_data.csv')
    source_features = source_data.iloc[:55705, 0:11].values
    source_target = source_data.iloc[:55705, 12].values

    target_data = pd.read_csv('F:/work_github/cof_data.csv')
    target_all_features = target_data.iloc[:6200, 0:11].values
    target_all_target = pd.to_numeric(target_data.iloc[:6200, 12], errors='coerce').values

    n_source_train = len(source_features)
    n_target_train = 2500
    n_target_test = len(target_all_features) - 2500

    indices = np.random.permutation(len(target_all_features))
    train_indices = indices[:n_target_train]
    test_indices = indices[n_target_train:]

    target_features_train = target_all_features[train_indices]
    target_target_train = target_all_target[train_indices]
    target_features_test = target_all_features[test_indices]
    target_target_test = target_all_target[test_indices]

    x_train = np.vstack((source_features, target_features_train))
    y_train = np.hstack((source_target, target_target_train))
    sample_size = [n_source_train, n_target_train]

    target_features = target_features_test
    target_target = target_target_test

    R2_aoc = 0
    save_ = True

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_xgbr, n_trials=30)

    print('Best hyperparameters: ', study.best_params)
    print('Best score: ', study.best_value)

    best_model = joblib.load('best_tradaboost_xgbr_model.pkl')
    y_pred = best_model.predict(target_features)

    target_features = pd.DataFrame(target_features)
    target_features['target'] = target_target
    target_features['prediction'] = y_pred
    target_features.to_csv('F:/work_github/xbgr/xgbr_tl.csv', index=False, encoding='utf-8')