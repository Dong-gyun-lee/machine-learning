import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from typing import Dict,Tuple,Union,List
from tqdm import tqdm
import time
from sklearn.ensemble import RandomForestRegressor

def kfold_rf(
    params: Dict[str, Union[int, float, str]],
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.array, np.array]:
    
    start = time.time()

    folds = KFold(n_splits=n_fold, shuffle=True, random_state=40)
    splits = folds.split(X, y)
    rf_oof = np.zeros((X.shape[0],))
    rf_preds = np.zeros((X_test.shape[0],))

    for fold, (train_idx, valid_idx) in tqdm(enumerate(splits)):
        print(f"============ Fold {fold} ============\n")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        X_test = X_test[X_train.columns]

        model = RandomForestRegressor(**params)
        model.fit(X_train,y_train)

        rf_oof[valid_idx] = model.predict(X_valid)
        rf_preds += model.predict(X_test)
        print(f"root_mean_squared_error: {np.sqrt(mean_squared_error(y_valid, rf_oof[valid_idx])):.5f}")
        print(f"mean_absolute_error: {mean_absolute_error(y_valid, rf_oof[valid_idx]):.5f}")
    
    rf_preds = rf_preds/n_fold
    rmse = np.sqrt(mean_squared_error(y,rf_oof))
    mae = mean_absolute_error(y, rf_oof)
    print(f"\n========================\n")
    print(f"rmse: {rmse:.5f}")
    print(f"mae: {mae:.5f}")
    
    end = time.time()
    print(f"{end - start:.5f} sec")
    
    return rf_oof, rf_preds

best_params_rf={'n_estimators': 12000,
 'max_depth': 18,
 'min_samples_split': 6,
 'min_samples_leaf': 3,
 'max_samples': 0.8,
 'max_features': 'auto',
 'random_state': 40,
 'n_jobs': -1}

rf_oof, rf_preds = kfold_rf(best_params_rf, 10, x_train, y_train, x_test)