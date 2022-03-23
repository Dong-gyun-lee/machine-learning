import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from typing import Dict,Tuple,Union,List
from catboost import CatBoostRegressor
from tqdm import tqdm
import time

from catboost import CatBoostRegressor
def kfold_cat(
    params: Dict[str, Union[int, float, str]],
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.array, np.array]:
    
    start=time.time()
    
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=40)
    splits = folds.split(X, y)
    cat_oof = np.zeros((X.shape[0],))
    cat_preds = np.zeros((X_test.shape[0],))

    for fold, (train_idx, valid_idx) in tqdm(enumerate(splits)):
        print(f"============ Fold {fold} ============\n")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        X_test = X_test[X_train.columns]

        model = CatBoostRegressor(**params)
        model.fit(
            X_train,y_train,
            early_stopping_rounds=100,
            cat_features=[0,1,2,3,4,5,6,7,8,9,10,14,17,19,22,23,25],
            verbose=100
        )

        cat_oof[valid_idx] = model.predict(X_valid)
        cat_preds += model.predict(X_test)
        print(f"root_mean_squared_error: {np.sqrt(mean_squared_error(y_valid, cat_oof[valid_idx])):.5f}")
        print(f"mean_absolute_error: {mean_absolute_error(y_valid, cat_oof[valid_idx]):.5f}")
    
    cat_preds = cat_preds/n_fold
    rmse = np.sqrt(mean_squared_error(y,cat_oof))
    mae = mean_absolute_error(y, cat_oof)
    
    print(f"\n========================\n")
    print(f"rmse: {rmse:.5f}")
    print(f"mae: {mae:.5f}")
    
    end = time.time()
    print(f"{end - start:.5f} sec")
    
    return cat_oof, cat_preds

best_params_cat = {'iterations': 4000, 
                   'colsample_bylevel': 0.8, 
                   'learning_rate': 0.1, 
                   'depth': 12,
                   'boosting_type': 'Ordered', #'Plain', 
                   'bagging_temperature': 2, 
                   'random_strength': 22, 
                   'od_type': 'IncToDec', 
                   'random_state':40}

cat_oof, cat_preds = kfold_cat(best_params_cat, 10, x_train, y_train, x_test)
