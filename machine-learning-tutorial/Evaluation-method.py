#%% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
# %%
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

#
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return(df)
# %%
titanic_df = pd.read_csv(r'C:\Users\uos\Desktop\mc\data\titanic_train.csv')
titanic_df=transform_features(titanic_df)

y_titanic = titanic_df['Survived']
x_titanic = titanic_df.drop('Survived',axis=1)

x_train,x_test,y_train,y_test = train_test_split(x_titanic,y_titanic,test_size=0.2,random_state=11)

# %% accuracy, precision, recall, confusion matrix 함수
def clf_evaluation(y_test,pred):
    confusion = confusion_matrix(y_test,pred)
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    print('ConfusionMatrix:')
    print(confusion)
    print('accuracy: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}'.format(accuracy,precision,recall))
# %%
lr_clf = LogisticRegression()
lr_clf.fit(x_train,y_train)
pred = lr_clf.predict(x_test)
clf_evaluation(y_test,pred)

# %% 예측될 확률 반환
proba = lr_clf.predict_proba(x_test)
print(proba[:5])

proba_result = np.concatenate([proba,pred.reshape(-1,1)],axis=1)
print(proba_result[:5])
# %% Binarizer
custom_threshold = 0.5
proba_1 = proba[:,1].reshape(-1,1)

# Binarizer: 개별원소들이 threshold보다 같거나 작으면 0, 크면 1을 반환
binarizer = Binarizer(threshold=custom_threshold).fit(proba_1)
custom_predict = binarizer.transform(proba_1)
clf_evaluation(y_test,custom_predict)
#%% 임계값 낮추면, precision down, recall up
custom_threshold = 0.4
proba_1 = proba[:,1].reshape(-1,1)

binarizer = Binarizer(threshold=custom_threshold).fit(proba_1)
custom_predict = binarizer.transform(proba_1)
clf_evaluation(y_test,custom_predict)

# %% 임계값 tuning
thresholds = [0.4,0.45,0.5,0.55,0.6]
def eval_thresholds(y_test,pred_proba_1,thresholds):
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1)
        custom_predict = binarizer.transform(pred_proba_1)
        print('\nthreshold:',custom_threshold)
        clf_evaluation(y_test,custom_predict)
        
eval_thresholds(y_test,proba[:,1].reshape(-1,1),thresholds)
# %% 임계값에 따른 변화 나타내기

# 레이블 값이 1이라고 예측할 확률을 추출
pred_proba_class1= lr_clf.predict_proba(x_test)[:,1]

#임계값(일반적으로 0.11~0.95정도) 별 precision, recall 계산해줌
precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_class1) 

# 임계값 10개 추출
thr_index = np.arange(0,thresholds.shape[0],15)
print('sample thresholds:',np.round(thresholds[thr_index],2))
# %%
def precision_recall_curve_plot(y_test,pred_proba1):
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba1) 
    
    plt.figure(figsize=(8,6))
    thresholds_boundary = thresholds.shape[0]
    plt.plot(thresholds,precisions[0:thresholds_boundary],linestyle='--',color='red',label='precision')
    plt.plot(thresholds,recalls[0:thresholds_boundary],label='recall')
    
    #threshold 값 X축의 Scale을 0.1 단위로 변경
    start,end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    
    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall')
    plt.legend(); plt.grid()
    plt.show()
# %%
precision_recall_curve_plot(y_test,lr_clf.predict_proba(x_test)[:,1])
# %% F1 Score (= recall과 precision의 조화평균)
f1 = f1_score(y_test,pred)
print('F1 score:{0:.4f}'.format(f1))
# %% clf_evaluation 함수에 F1 추가
def clf_evaluation2(y_test,pred):
    confusion = confusion_matrix(y_test,pred)
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    print('ConfusionMatrix:')
    print(confusion)
    print('accuracy: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, F1: {3:.4f}'.format(accuracy,precision,recall,f1))

def eval_thresholds2(y_test,pred_proba_1,thresholds):
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1)
        custom_predict = binarizer.transform(pred_proba_1)
        print('\nthreshold:',custom_threshold)
        clf_evaluation2(y_test,custom_predict)
# %%
thresholds = [0.4,0.45,0.5,0.55,0.6]
pred_proba_1 = lr_clf.predict_proba(x_test)[:,1].reshape(-1,1)
eval_thresholds2(y_test,pred_proba_1,thresholds)
# %% ROC Curve & AUC
pred_proba_class1= lr_clf.predict_proba(x_test)[:,1]
fprs, tprs, threshold = roc_curve(y_test,pred_proba_class1)
thr_index = np.arange(1,threshold.shape[0],5)
print('sample threshold:',np.round(threshold[thr_index],2))
print('sample fpr:',np.round(fprs[thr_index],3))
print('sample tpr:',np.round(tprs[thr_index],3))
# %%
def roc_curve_plot(y_test,pred_proba_c1):
    fprs, tprs, threshold = roc_curve(y_test,pred_proba_c1)
    #ROC Curve
    plt.plot(fprs,tprs,label='ROC Curve')
    #가운데 대각선 직선 그림
    plt.plot([0,1],[0,1],'k--',label='Random')
    
    #FPR X축의 Scale을 0.1 단위로 변경, X,Y축 명 설정
    start,end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    
    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('FPR( 1 - Sensitivity )'); plt.ylabel('TPR( Recall )')
    plt.legend(); plt.grid()
    plt.show()
#%%
roc_curve_plot(y_test,pred_proba_class1)
# %% AUC Score
roc_score = roc_auc_score(y_test,pred_proba_class1)
print('ROC AUC 값: {0:.4f}'.format(roc_score))
# %% AUC Score 추가
def clf_evaluation3(y_test,pred,pred_proba):
    confusion = confusion_matrix(y_test,pred)
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    roc_score = roc_auc_score(y_test,pred_proba)
    print('ConfusionMatrix:')
    print(confusion)
    print('accuracy: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, F1: {3:.4f}, ROC_AUC: {4:.4f}'.format(accuracy,precision,recall,f1,roc_score))

def eval_thresholds3(y_test,pred_proba_1,thresholds):
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1)
        custom_predict = binarizer.transform(pred_proba_1)
        print('\nthreshold:',custom_threshold)
        clf_evaluation3(y_test,custom_predict,pred_proba_1)
# %%
thresholds = [0.4,0.45,0.5,0.55,0.6]
pred_proba_1 = lr_clf.predict_proba(x_test)[:,1].reshape(-1,1)
eval_thresholds3(y_test,pred_proba_1,thresholds)
# %%
