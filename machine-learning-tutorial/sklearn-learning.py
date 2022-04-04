!pip install scikit-learn
#%% iris data 실습
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# %% 자료파악
iris = load_iris()
iris_data = iris.data
iris_label = iris.target
iris.target_names
iris_df = pd.DataFrame(data = iris_data, columns=iris.feature_names)
iris_df.head(3)
iris_df.info()
iris_df.describe()
# %% 데이터 세트 분리
x_train,x_test,y_train,y_test = train_test_split(iris_data,iris_label,test_size=0.2,random_state=11)
# %% 모델학습 및 예측
dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(x_train,y_train)
pred = dt_clf.predict(x_test)
pred
# %% Accuracy 평가
from sklearn.metrics import accuracy_score
print('예측 정확도:{0:.4f}'.format(accuracy_score(y_test,pred)))
# %% K fold CV
features = iris.data
labels = iris.target
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)
cv_accuracy = []
dt_clf = DecisionTreeClassifier(random_state=156)

n_iter=0
for train_index, test_index in kfold.split(features):
    x_train,x_test = features[train_index], features[test_index]
    y_train,y_test = labels[train_index], labels[test_index]
    
    # 모델 학습 및 예측
    dt_clf.fit(x_train,y_train)
    pred = dt_clf.predict(x_test)
    n_iter += 1
    
    # 반복 시마다 정확도 측정
    accuracy = np.round(accuracy_score(y_test,pred),4)
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    print('\n## Accuracy:{0}, train_size:{1}, test_size:{2}'.format(accuracy,train_size,test_size))
    print('{0} test index:{1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)

print('\n### average Acurracy:{0}'.format(np.mean(cv_accuracy)))
# %% Stratified K fold (층화추출법 사용)
features = iris.data
labels = iris.target 
cv_accuracy2 = []

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3)

n_iter = 0
for train_index, test_index in skf.split(features,labels):
    x_train,x_test = features[train_index], features[test_index]
    y_train,y_test = labels[train_index], labels[test_index]
    
    # 모델 학습 및 예측
    dt_clf.fit(x_train,y_train)
    pred = dt_clf.predict(x_test)
    n_iter += 1
    
    # 반복 시마다 정확도 측정
    accuracy = np.round(accuracy_score(y_test,pred),4)
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    print('\n## Accuracy:{0}, train_size:{1}, test_size:{2}'.format(accuracy,train_size,test_size))
    print('{0} test index:{1}'.format(n_iter,test_index))
    cv_accuracy2.append(accuracy)


print('\n### average Acurracy:{0}'.format(np.mean(cv_accuracy2)))
# %% 간단한 함수 이용한 k fold cv
# classifier는 Stratified K fold 방식으로 분할해줌
# Regressor는 k fold로 분할
from sklearn.model_selection import cross_val_score, cross_validate
scores = cross_val_score(dt_clf,features,labels,scoring='accuracy',cv=3)
print('교차 검증별 정확도:',np.round(scores,4))
# %% Gridresearchcv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

x_train,x_test,y_train,y_test = train_test_split(iris_data,iris_label,test_size=0.2,random_state=11)
dt_clf = DecisionTreeClassifier(random_state=156)
grid_parameter = {'max_depth':[1,2,3],'min_samples_split':[2,3]}
grid_dtree = GridSearchCV(dt_clf,param_grid=grid_parameter,cv=3,refit=True)
grid_dtree.fit(x_train,y_train)
scores_df = pd.DataFrame(grid_dtree.cv_results_)
scores_df[['params','mean_test_score','rank_test_score','split0_test_score','split1_test_score','split2_test_score']]

# 정확도 가장 좋은 것 나타내기 
print('best parameter:',grid_dtree.best_params_)
print('best accuracy:{0:.4f}'.format(grid_dtree.best_score_))

# 
estimator = grid_dtree.best_estimator_
pred = estimator.predict(x_test)
print('test accuracy: {0:.4f}'.format(accuracy_score(y_test,pred)))
# %%
