#%% Optuna 패키지를 이용한 하이퍼 파라미터 튜닝 (K-fold cross validation) 
# param에는 각자 모델에 맞는 parameter를 넣으면 된다.

import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

# XGBoost
def objectiveXGB(trial: Trial, X, y,n_fold):
    param = {
        "n_estimators" : trial.suggest_int('n_estimators', 500, 2000),
        'max_depth':trial.suggest_int('max_depth', 8, 20),
        'min_child_weight':trial.suggest_int('min_child_weight', 1, 20),
        'gamma':trial.suggest_int('gamma', 0.5, 3),
        'learning_rate': 0.01,
        'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.5, 1, 0.1),
        'nthread' : -1,
        'tree_method': 'hist',
        'predictor': 'cpu_predictor',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample': trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0] ),
        'random_state': 40
    }
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    splits = folds.split(X, y)
    
    xgb_oof = np.zeros((X.shape[0],))

    for fold, (train_idx, valid_idx) in enumerate(splits):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBRegressor(**param)
        model.fit(
            X_train,y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=100,
        )

        xgb_oof[valid_idx] = model.predict(X_valid)
        
    score = mean_absolute_error(y, xgb_oof)

    return score

study = optuna.create_study(direction='minimize',sampler=TPESampler())
study.optimize(lambda trial : objectiveXGB(trial, x_train,  y_train,10), n_trials=50)
print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

#%%
# Optuna 패키지를 이용한 하이퍼 파라미터 튜닝 (Hold-out) 

def objectiveXGB(trial: Trial, X, y, test):
    param = {
        "n_estimators" : trial.suggest_int('n_estimators', 500, 2000),
        'max_depth':trial.suggest_int('max_depth', 8, 20),
        'min_child_weight':trial.suggest_int('min_child_weight', 1, 20),
        'gamma':trial.suggest_int('gamma', 0.5, 3),
        'learning_rate': 0.01,
        'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.5, 1, 0.1),
        'nthread' : -1,
        'tree_method': 'hist',
        'predictor': 'cpu_predictor',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample': trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0] ),
        'random_state': 40
    }
    y = np.array(y_train)
    X_train, X_test, y_train, y_test = train_test_split(X, y.flatten(), test_size=0.1) #y.flatten은 y를 array로 했을 때 해줌
    
    y_train = y_train.reshape(-1, 1)
    y_test  = y_test.reshape(-1, 1)

    model = XGBRegressor(**param)
    xgb_model = model.fit(X_train, y_train, verbose=False, eval_set=[(X_test, y_test)])
    score = mean_squared_error(xgb_model.predict(X_test), y_test, squared=False)

    return score

study = optuna.create_study(direction='minimize',sampler=TPESampler())
study.optimize(lambda trial : objectiveXGB(trial, x_train,  y_train, x_test), n_trials=50)
print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

best_param = study.best_trial.params


#%%  Random Forest
def objectiveRF(trial: Trial, X, y):
    param = {
        "n_estimators" : trial.suggest_int('n_estimators', 500, 2000),
        'max_depth':trial.suggest_int('max_depth', 8, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 6, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 32),
        'max_features':'auto',
        'n_jobs':-1,
        'random_state':40,
        'max_samples': trial.suggest_categorical('max_samples', [0.6,0.7,0.8,0.9] )
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(**param)
    rf_model = model.fit(X_train, y_train)
    score = mean_squared_error(rf_model.predict(X_test), y_test)

    return score

study = optuna.create_study(direction='minimize',sampler=TPESampler())
study.optimize(lambda trial : objectiveRF(trial, x_train,  y_train, x_test), n_trials=50)
print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))
#%%
def objectiveCat(trial,X,y):
    param = {
        'iterations' : trial.suggest_int('iterations', 100, 3000),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
        'random_strength' :trial.suggest_int('random_strength', 0, 100),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        "used_ram_limit": "3gb",
        'eval_metric':'RMSE',
        'random_state':40
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = CatBoostRegressor(**param)

    model.fit(X_train,y_train, verbose=100, cat_features=[0,1,2,3,4,5] ,early_stopping_rounds=100)

    preds = model.predict(X_test)
    score = mean_squared_error(y_test, preds)
    return score

study = optuna.create_study(direction='minimize',sampler=TPESampler())
study.optimize(lambda trial : objectiveCat(trial, x_train,  y_train, x_test), n_trials=50)
print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))
