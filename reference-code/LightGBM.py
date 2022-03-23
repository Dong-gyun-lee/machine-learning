import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from typing import Dict,Tuple,Union,List
from tqdm import tqdm
import time

def kfold_lgb(
    params: Dict[str, Union[int, float, str]],
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.array, np.array]:
    
    start = time.time()

    folds = KFold(n_splits=n_fold, shuffle=True, random_state=40)
    splits = folds.split(X, y)
    lgb_oof = np.zeros((X.shape[0],))
    lgb_preds = np.zeros((X_test.shape[0],))

    for fold, (train_idx, valid_idx) in tqdm(enumerate(splits)):
        print(f"============ Fold {fold} ============\n")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        X_test = X_test[X_train.columns]

        model = LGBMRegressor(**params)
        model.fit(
            X_train,y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=100
        )

        lgb_oof[valid_idx] = model.predict(X_valid)
        lgb_preds += model.predict(X_test)
        print(f"root_mean_squared_error: {np.sqrt(mean_squared_error(y_valid, lgb_oof[valid_idx])):.5f}")
        print(f"mean_absolute_error: {mean_absolute_error(y_valid, lgb_oof[valid_idx]):.5f}")
    
    lgb_preds = lgb_preds/n_fold
    rmse = np.sqrt(mean_squared_error(y,lgb_oof))
    mae = mean_absolute_error(y, lgb_oof)
    print(f"\n========================\n")
    print(f"rmse: {rmse:.5f}")
    print(f"mae: {mae:.5f}")
    n_fold: int,
    
    end = time.time()
    print(f"{end - start:.5f} sec")
    
    return lgb_oof, lgb_preds

best_params_lgb={'n_estimators': 12000,
                 'max_depth': 18,
                 'learning_rate': 0.08,
                  'reg_lambda':0,
                 'min_child_weight': 10,
                 'subsample': 0.8,
                 'objective': 'regression',
                 'boosting_type': 'goss',
                 'colsample_bytree': 0.8,
                 'metric': 'rmse',
                  'num_leaves':20000,
                  'min_data_in_leaf': 3,
                 'random_state': 40}

lgb_oof_l lgb_preds = kfold_lgb(best_params_lgb, 10, x_train, y_train, x_test)