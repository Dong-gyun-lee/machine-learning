import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from typing import Dict, Tuple, Union, List


def kfold_xgb(
    params: Dict[str, Union[int, float, str]],
    n_fold: int,
    X: np.array,
    y: pd.DataFrame,
    X_test: np.array,
) -> Tuple[np.array, np.array]:

    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    splits = folds.split(X, y)
    xgb_oof = np.zeros((X.shape[0],))
    xgb_preds = np.zeros((X_test.shape[0],))

    for fold, (train_idx, valid_idx) in enumerate(splits):
        print(f"============ Fold {fold} ============\n")
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBRegressor(**params)
        model.fit(
            X_train,y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=100,
        )

        xgb_oof[valid_idx] = model.predict(X_valid)
        xgb_preds += model.predict(X_test)
        print(f"Mean_absolute_error: {mean_absolute_error(y_valid, xgb_oof[valid_idx]):.5f}")
    
    xgb_preds = xgb_preds/n_fold
    mean_absolute_error_score = mean_absolute_error(y, xgb_oof)
    print(f"\n========================\n")
    print(f"Mean_absolute_error Score: {mean_absolute_error_score:.5f}")

    return xgb_oof, xgb_preds

best_params = {'n_estimators': 1407,
 'max_depth': 18,
 'min_child_weight': 15,
 'gamma': 0,
 'colsample_bytree': 0.8,
 'lambda': 3.931968791423167,
 'alpha': 0.0018831728905603729,
 'subsample': 0.7,
 'nthread': -1,
 'learning_rate': 0.01,
 'random_state': 40}

xgb_oof, xgb_preds = kfold_xgb(best_params, 10, x_train, y_train, x_test)