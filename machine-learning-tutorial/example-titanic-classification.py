#%%
from inspect import Parameter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
# %% 전처리

def fillna(df):
    df['Age'].fillna(titanic_df['Age'].mean(),inplace=True)
    df['Cabin'].fillna('N',inplace=True)
    df['Embarked'].fillna('N',inplace=True)
    df['Fare'].fillna(0,inplace=True)
    return df

def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
    return df

#label encoding
def format_features(df):
    df['Cabin']=df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 합치기
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return(df)
# %%
titanic_df = pd.read_csv(r'C:\Users\uos\Desktop\mc\data\titanic_train.csv')
titanic_df=transform_features(titanic_df)
titanic_df.head(5)

y_titanic = titanic_df['Survived']
x_titanic = titanic_df.drop('Survived',axis=1)

x_train,x_test,y_train,y_test = train_test_split(x_titanic,y_titanic,test_size=0.2,random_state=11)
# %% 모델
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression()

#DecisionTree
dt_clf.fit(x_train,y_train)
dt_pred = dt_clf.predict(x_test)
print('DecisionTreeClassifer Accuracy:{0:.4f}'.format(accuracy_score(y_test,dt_pred)))

#RandomForestClassifier
rf_clf.fit(x_train,y_train)
rf_pred = rf_clf.predict(x_test)
print('RandomForestClassifer Accuracy:{0:.4f}'.format(accuracy_score(y_test,rf_pred)))

#LogisticRegression
lr_clf.fit(x_train,y_train)
lr_pred = lr_clf.predict(x_test)
print('LogisticRegression Accuracy:{0:.4f}'.format(accuracy_score(y_test,lr_pred)))
# %% K fold CV
def exec_kfold(clf,folds=5):
    kfold = KFold(n_splits=folds)
    scores = []
    
    for iter_count, (train_index, test_index) in enumerate(kfold.split(x_titanic)):
        x_train,x_test = x_titanic.values[train_index], x_titanic.values[test_index]
        y_train,y_test = y_titanic.values[train_index], y_titanic.values[test_index]
    
        # 모델 학습 및 예측
        clf.fit(x_train,y_train)
        pred = clf.predict(x_test)
    
        # 반복 시마다 정확도 측정
        accuracy = np.round(accuracy_score(y_test,pred),4)
        print('{0} accuracy:{1}'.format(iter_count,accuracy))
        scores.append(accuracy)
    mean_score = np.mean(scores)
    print('mean_accuracy:{0:.4f}'.format(mean_score))
    
exec_kfold(dt_clf,folds=5)
# %% cross_val_score 이용한 k fold CV(labeling이 되어있어서, Stratified K fold CV로 나옴)
score = cross_val_score(dt_clf,x_titanic,y_titanic,cv=5)
for iter_count, accuracy in enumerate(score):
    print('iter:{0}, accuracy:{1:.4f}'.format(iter_count,accuracy))
print('mean_accuracy:{0:.4f}'.format(np.mean(score)))
# %%
parameters = {'max_depth':[2,3,5,10],'min_samples_split':[2,3,5],'min_samples_leaf':[1,5,8]}
grid_dclf = GridSearchCV(dt_clf,param_grid=parameters,scoring='accuracy',cv=5)
grid_dclf.fit(x_train,y_train)

print('best parameter:',grid_dclf.best_params_)
print('best accuracy: {0:.4f}'.format(grid_dclf.best_score_))
best_dclf = grid_dclf.best_estimator_

prediction = best_dclf.predict(x_test)
accuracy = accuracy_score(y_test,prediction)
print('Accuracy:{0:.4f}'.format(accuracy))
# %%
