#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
from sklearn.metrics import f1_score,roc_auc_score,precision_recall_curve,roc_curve
from sklearn.linear_model import LogisticRegression
#%% 데이터 확인
diabetes = pd.read_csv(r'C:\Users\uos\Desktop\mc\data\diabetes.csv')
diabetes.info()
diabetes['Outcome'].value_counts()

#%% 데이터 전처리
x = diabetes.iloc[:,:-1]
y = diabetes.iloc[:,-1]

scaler = StandardScaler()
x_scale = scaler.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_scale,y,test_size=0.2,random_state=11)

lr_clf = LogisticRegression()
lr_clf.fit(x_train, y_train)
pred = lr_clf.predict(x_test)
pred_proba = lr_clf.predict_proba(x_test)[:,1]
# %%
def clf_evaluation3(y_test,pred):
    confusion = confusion_matrix(y_test,pred)
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    roc_score = roc_auc_score(y_test,pred)
    print('ConfusionMatrix:')
    print(confusion)
    print('accuracy: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, F1: {3:.4f}, ROC_AUC: {4:.4f}'.format(accuracy,precision,recall,f1,roc_score))

def eval_thresholds3(y_test,pred_proba_1,thresholds):
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1)
        custom_predict = binarizer.transform(pred_proba_1)
        print('\nthreshold:',custom_threshold)
        clf_evaluation3(y_test,custom_predict)
# %%
clf_evaluation3(y_test,pred)
threshold = [0.3,0.33,0.36,0.39,0.42,0.45,0.48,0.5]
eval_thresholds3(y_test,pred_proba.reshape(-1,1),threshold)
# %%
