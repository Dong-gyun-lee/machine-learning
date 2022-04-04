#%%
### Data enconding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#%%
# Label encoding
# 숫자로 labeling되므로 회귀 문제에서는 이러한 인코딩을 적용하지 않음
items = ['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print('encoding:',labels)
print('encoding classes:',encoder.classes_) #encoding된 순서대로 나옴
print('decoding:',encoder.inverse_transform([4,5,3,5,2,1,4])) #decoding 뭔지 나옴
# %% One-Hot encoding
items = ['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
labels = labels.reshape(-1,1)

oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_label = oh_encoder.transform(labels)
print(oh_label.toarray())
print(oh_label.shape)
# %% pandas 이용한 one hot encoding
df = pd.DataFrame({'item':['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']})
pd.get_dummies(df)
# %% feature scaling & normalization
# StandardScaler ~ N(0,1)
iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data,columns=iris.feature_names) #list(iris_df.columns) = iris.feature_names
print(iris_df.mean())
print(iris_df.var())

scaler = StandardScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)
iris_df_scaled = pd.DataFrame(iris_scaled,columns=iris.feature_names)
iris_df_scaled.head(5)
print(iris_df_scaled.mean())
print(iris_df_scaled.var())

# %% MinMaxScaler (0~1값 가짐, 음수 값이 있으면 -1~1값 가짐)
# 데이터의 분포가 가우시안 분포가 아닐 때 사용?

scaler2 = MinMaxScaler()
scaler2.fit(iris_df)
iris_scaled2 = scaler2.transform(iris_df)
iris_scaled2 = pd.DataFrame(iris_scaled2,columns=iris.feature_names) 
iris_scaled2.head(5)

## Scale할 때 유의할 점

# 1.전체 DATASET을 Scale한 뒤 분리하는 것이 제일 좋음
# 2.train set으로 fit한 scaler로 test set을 transform해줘야  scale이 맞음
# %% 
### titanic 예제
titanic_df = pd.read_csv(r'C:\Users\uos\Desktop\mc\data\titanic_train.csv')
print('\n### 학습 데이터 정보 ###\n')
print(titanic_df.info())
titanic_df.head(3)

#결측치 처리
titanic_df['Age'].fillna(titanic_df['Age'].mean(),inplace=True)
titanic_df['Cabin'].fillna('N',inplace=True)
titanic_df['Embarked'].fillna('N',inplace=True)
titanic_df.isna().sum().sum()

# 문자열 feature 살펴보기
print('\n # Sex 분포:\n',titanic_df['Sex'].value_counts())
print('\n # Cabin 분포:\n',titanic_df['Cabin'].value_counts())
print('\n # Embarked 분포:\n',titanic_df['Embarked'].value_counts())

# Cabin은 앞문자만 추출해서 정리
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
titanic_df['Cabin'].value_counts()

#%% 반응변수 시각화해서 확인해보기
titanic_df.groupby(['Sex','Survived'])['Survived'].count()
sns.barplot(x='Sex',y='Survived',data=titanic_df) 
sns.barplot(x='Pclass',y='Survived',hue='Sex',data=titanic_df) #3개 고려

# 함수 사용
def get_category(age):
    cat = ''
    if age <=5: cat='baby'
    elif age<=12: cat='child'
    elif age<=18: cat='teenager'
    elif age<=25: cat = 'student'
    elif age<=35: cat = 'young Adult'
    elif age<=60: cat = 'adult'
    else: cat='elderly'
    return cat

group_name=['baby','child','teenager','student','young Adult','adult','elderly']
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x:get_category(x))
titanic_df[['Age','Age_cat']].head(10)
plt.figure(figsize=(10,6))
sns.barplot(x='Age_cat',y='Survived',hue='Sex',data=titanic_df,order=group_name) #order를 group_name 순서대로
titanic_df.drop('Age_cat',axis=1,inplace=True) #다시 없애줌

# %% 문자열을 숫자형으로 변환
def encode_features(dataDF):
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])
    return dataDF

titanic_df = encode_features(titanic_df)
titanic_df.head()