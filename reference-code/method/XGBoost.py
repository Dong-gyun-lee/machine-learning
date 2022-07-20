import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from typing import Dict,Tuple,Union,List
from tqdm import tqdm
import time

def kfold_xgb(
    params: Dict[str, Union[int, float, str]],
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: np.array,
) -> Tuple[np.array, np.array]:
    
    start = time.time()

    folds = KFold(n_splits=n_fold, shuffle=True, random_state=40)
    splits = folds.split(X, y)
    xgb_oof = np.zeros((X.shape[0],))
    xgb_preds = np.zeros((X_test.shape[0],))

    for fold, (train_idx, valid_idx) in tqdm(enumerate(splits)):
        print(f"============ Fold {fold} ============\n")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        X_test = X_test[X_train.columns]

        model = XGBRegressor(**params)
        model.fit(
            X_train,y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=100,
        )

        xgb_oof[valid_idx] = model.predict(X_valid)
        xgb_preds += model.predict(X_test)
        print(f"root_mean_squared_error: {np.sqrt(mean_squared_error(y_valid, xgb_oof[valid_idx])):.5f}")
        print(f"Mean_absolute_error: {mean_absolute_error(y_valid, xgb_oof[valid_idx]):.5f}")
    
    xgb_preds = xgb_preds/n_fold
    rmse = np.sqrt(mean_squared_error(y,xgb_oof))
    mae = mean_absolute_error(y, xgb_oof)
    print(f"\n========================\n")
    print(f"RMSE: {rmse:.5f}")
    print(f"Mean_absolute_error Score: {mae:.5f}")
    
    end = time.time()
    
    print(f"{end - start:.5f} sec")

    return xgb_oof, xgb_preds

best_params_xgb = {'n_estimators': 10000,
 'max_depth': 18,
 'min_child_weight': 10,
 'gamma': 0,
 'colsample_bytree': 0.8,
 'lambda': 0.7,
 'alpha': 0.1,
 'subsample': 1,
 'nthread': -1,
 'learning_rate': 0.08,
 'random_state': 40}

xgb_oof, xgb_preds = kfold_xgb(best_params_xgb, 10, x_train, y_train, x_test)
