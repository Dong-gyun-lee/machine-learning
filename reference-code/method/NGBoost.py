
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from typing import Dict,Tuple,Union,List
from tqdm import tqdm
import time
from ngboost import NGBRegressor

def kfold_ngb(
    params: Dict[str, Union[int, float, str]],
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.array, np.array]:
    
    start = time.time()
    
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=40)
    splits = folds.split(X, y)
    ngb_oof = np.zeros((X.shape[0],))
    ngb_preds = np.zeros((X_test.shape[0],))

    for fold, (train_idx, valid_idx) in enumerate(splits):
        print(f"============ Fold {fold} ============\n")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        X_test = X_test[X_train.columns]

        model = NGBRegressor(**params)
        model.fit(X_train,y_train,X_valid,y_valid,early_stopping_rounds=100)

        ngb_oof[valid_idx] = model.predict(X_valid)
        ngb_preds += model.predict(X_test)
        print(f"root_mean_squared_error: {np.sqrt(mean_squared_error(y_valid, ngb_oof[valid_idx])):.5f}")
        print(f"mean_absolute_error: {mean_absolute_error(y_valid, ngb_oof[valid_idx]):.5f}")
        
    ngb_preds = ngb_preds/n_fold
    root_mean_squared_error_score = np.sqrt(mean_squared_error(y, ngb_oof))
    mean_absolute_error_score = mean_absolute_error(y, ngb_oof)
    print(f"\n========================\n")
    print(f"root_mean_squared_error Score: {root_mean_squared_error_score:.5f}")
    print(f"root_mean_squared_error Score: {mean_absolute_error_score:.5f}")
    
    end = time.time()
    print(f"{end-start:.5f}sec")
    
    return ngb_oof, ngb_preds

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

default_tree_learner2 = DecisionTreeRegressor(
    criterion="friedman_mse",
    min_samples_split=10,
    min_samples_leaf=5,
    min_weight_fraction_leaf=0.05,
    max_depth=15,
    splitter="best",
    random_state=40,
)

default_linear_learner2 = Ridge(alpha=0.0, random_state=40)

best_params_ngb = {'n_estimators':25000,'natural_gradient':True,'learning_rate':0.08,'verbose_eval':100,
            'random_state':40,'minibatch_frac':0.8,'col_sample':0.8,'Base':default_tree_learner2}

ngb_oof, ngb_preds = kfold_ngb(best_params_ngb, 10, x_train, y_train, x_test)
