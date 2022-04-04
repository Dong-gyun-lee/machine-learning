#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% 파일 불러오기
data = pd.read_excel(r'경로\파일.csv')
data = pd.read_csv(r'경로\파일.csv',encoding='cp949')

#%% 데이터 기본 정보 파악

data.head() # 데이터 상위 일부 나타냄
data.info() # 데이터 정보
data.describe() # 데이터 요약 통계량
data.isnull().sum() #결측치 확인

data.column_name.value_counts() # column의 데이터 확인

# 데이터 필터링
data[(data.column_name=='관측값')|(data.column_name=='관측값')]
data[~((data.column_name==5)&(data.column_name.일>12))]

#%% 칼럼의 타입 바꾸기
data = data.astype({'column_name':'int'})

#칼럼명바꾸기
data.rename(columns={'기존칼럼명':'새칼럼명'},inplace=True)

# 데이터 관측값 변경
data.loc[data['column_name'] == '기존값', 'column_name'] = '새로운값' #int,float 속성도 가능
#%% 열,행 지우기
data.drop(['column_name'],axis=1,inplace=True) # 해당 column 지우기
data.dropna(axis=0,inplace=True) # na 값 가지는 row 지우기
data.drop_duplicates(inplace=True) #중복 row 제거

#index reset
data.reset_index(inplace=True,drop=True)

#%% 결측값 대체
data.fillna(0,inplace=True) #0으로 대체
data.fillna('missing',inplace=True) #'missing'으로 대체

data.fillna(method='ffill',inplace=True) # 앞 방향으로 대체
data.fillna(method='bfill',inplace=True) # 뒷 방향으로 대체

data.fillna(data.mean()), data.where(pd.notnull(data), data.mean(), axis='columns') #변수별 평균으로 대체
#%% 데이터 melt
data = pd.melt(data,id_vars=['나', '머', '지', '칼', '럼'],var_name='기존칼럼명',value_name='새칼럼명')

#%% groupby
data.groupby('column_name',as_index=False).sum()
data.groupby('column_name',as_index=False).mean()

#%% merge , concat
pd.merge(data1,data2,left_on='column_name1',right_on='column_name12',how='left') # column 명 기준으로 병합/ how는 left, outer 등
pd.concat([data1,data2],axis=0, ignore_index=True) # 단순 이어붙이기 , ignore_index 는 인덱스 재배열


#%% 이상치 제거

from collections import Counter

data_list = list(set(data.column_name))
def detect_outliers(data, data_list): 
    outlier_indices = []
    for col in data_list:
        data2 = data[data.column_name==col]
        Q1 = np.percentile(data2['column_name'], 25) 
        Q3 = np.percentile(data2['column_name'], 75) 
        IQR = Q3 - Q1 
        outlier_step = 1.5*IQR 
        outlier_list_col = data2[(data2['column_name'] < Q1 - outlier_step) | (data2['column_name'] > Q3 + outlier_step)].index 
        outlier_indices.extend(outlier_list_col) 
        print(outlier_indices)
    return outlier_indices
outlier_indices1 = detect_outliers(data, data_list)
outlier_indices1

ix=[i for i in data.index if i not in outlier_indices1]
data_not_outlier = data.loc[ix]

#%% 순서형 변수 만들기 예시

edu_order = {
    'Lower secondary' : 0,
    'Secondary / secondary special' : 1,
    'Incomplete higher' : 2,
    'Higher education' : 3,
    'Academic degree' : 4
}
data.edu_type = data.edu_type.map(edu_order)

# 더미변수 만들기
data_dummy= pd.get_dummies(data)

#%% 정규화
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

#%% 날짜 데이터 관리
data['날짜']=pd.to_datetime(data['날짜'])

data['년'] = data['날짜'].dt.year
data['월'] = data['날짜'].dt.month
data['일'] = data['날짜'].dt.day

#%% 공휴일 만들기
hol = '''
20220101
20220102
'''
hol
holiday = hol.split('\n')
holiday.pop(0)
holiday.pop(-1)
for i in range(len(holiday)) :
  holiday[i] = holiday[i][0:4] + '-' + holiday[i][4:6] + '-' + holiday[i][6:8]

data['날짜']=pd.to_datetime(data['날짜'])
data['요일']=data['날짜'].dt.weekday

data[data['날짜'].isin(holiday)]
data['휴일'] = 0
data.loc[data['날짜'].isin(holiday), '휴일'] = 1
data.loc[(data['요일']==5) | (data['요일']==6),'휴일'] = 1
data.reset_index(inplace=True,drop=True)

# 하루 전 날짜 대입
import datetime
for i in range(data.shape[0]):
        data.at[i,'날짜'] = data.at[i,'날짜']+datetime.timedelta(1)
data

#%% 뒤에 괄호가 있는 관측값들 괄호 없애기 
b1  = []
for i in range(data.shape[0]):
    if '(' in data.at[i,'column_name']:
        ind = str(data.at[i,'column_name']).index('(')
        nou = str(data.at[i,'column_name'])[0:ind]
        b1.append(nou)
    else:
        b1.append(data.at[i,'column_name'])
data['column_name'] = b1

#%% 기존 데이터 프레임 정보로 dictionary 생성해서 새로운 dataframe 만들기
new_dict = {}
for i in range(data.shape[0]):
    key = str(data['key_column'][i])
    value = str(data['value_column'][i])
    if (key in new_dict.keys()):
            new_dict[key].append(value)
    else :
        new_dict.setdefault(key,[])
        new_dict[key].append(value)
new_dict
new_dict.keys()
new_dict.values()
new_dict_frame = pd.DataFrame({'key_column': new_dict.keys(), 'value_list':new_dict.values()})
new_dict_frame

new_dict_number = []
for i in range(new_dict_frame.shape[0]):
    new_dict_number.append(len(new_dict_frame.at[i,'value_list']))