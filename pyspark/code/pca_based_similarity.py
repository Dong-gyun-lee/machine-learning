## spark 환경 만들기

import numpy as np
import pandas as pd
import time
from glob import glob
import findspark
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import seaborn as sns

appname = 'Pyspark - Seed - similarity'
master = 'local[20]'
findspark.init()
spark = SparkSession.builder\
.appName(appname).master(master).config('spark.driver.memory','10g')\
.config('spark.executor.memory','10g')\
.config('spark.ui.port','port 번호')\
.config('spark.local.dir', 'local 경로')\
.getOrCreate()

## seed data 불러오기
seed_id = spark.read.csv("seed_file 경로",header=True)

## 잘못 입력된 id 정제하는 작업
from pyspark.sql.functions import regexp_replace,concat
seed_id2 = seed_id.withColumn("id",F.when(F.col("adid").isNull(),F.col("_c1")).otherwise(F.col("adid"))) # null로 입력된 column 확인 후 데이터 정제
seed_id2 = seed_id2.select("id").withColumnRenamed("id","adid") #column 이름바꾸기
seed_oaid = seed_id2.withColumn("oaid",regexp_replace("adid","-",""))\ 
.withColumn("a",F.lit("a_")).select(concat(F.col("a"),F.col("oaid")).alias("oaid")) #id 변환 "-"없애고 "a_"단어붙히기
seed_oaid = seed_oaid.withColumn("seed",F.lit(1)) # 모든 값이 1인 column 생성

##  데이터 메모리 문제로 인해 cateogry 별로 filtering하여 join하고 저장 진행하는 작업
right_data = spark.read.orc("right_data 경로")
date = ["category_list 입력"] #category 값에 해당됨
for i in range(1,len(date)+1):
    globals()["join_data_part" + str(i)] = right_data.filter(F.col("category_column")==date[i-1])

for i in range(1,len(date)+1):
    globals()["join_data_part" + str(i)] = globals()["join_data_part" + str(i)]\
    .join(seed_oaid,on="oaid",how="left").na.fill(0,["seed"])

    
## join된 data 저장 : file이 크기 때문에 중간 중간 저장하고 다시 불러오는 것이 바람직하다.
for i in range(1,len(date)+1):
    globals()["join_data_part" + str(i)].\
    write.parquet("경로/join_data_part"+str(i))

## join된 data 불러오기
for i in range(1,len(date)+1):
    globals()["join_data_part" + str(i)] = \
    spark.read.parquet("경로/join_data_part"+str(i))
    
## 분할된 data union
join_data = join_data_part1
for i in range(2,len(date)+1):
    join_data = join_data.union(globals()["join_data_part1" + str(i)])

## 분석을 위한 feature column 만들기
from pyspark.ml.feature import VectorAssembler

featureCols=join_data.columns[2:-1] # column 중 feature로 만들 column 해당
assembler = VectorAssembler(inputCols=featureCols, outputCol = 'features',handleInvalid="keep")
data_assemble = assembler.transform(join_data)

## 분석을 위한 데이터 정리
data_feat = data_assemble.select("oaid","seed","features") # 사용할 변수 선택
data_feat = data_feat.filter((F.col("oaid").contains("i_"))|(F.col("oaid").contains("a_"))) # "oaid" column에서 특정 문자를 포함한 값만 사용
data_seed = data_feat.filter(F.col("seed")==1) #seed가 1인 data filtering

## seed data를 잘 표현하는 축으로 데이터 PCA변환
from pyspark.ml.feature import PCA

pca = PCA(k=50, inputCol="features") # K=50으로 설정, 70~90% 설명하는 주성분 개수 설정
pca.setOutputCol("pca_features") #output column을 "pca_feautures" 라고 설정
model = pca.fit(data_seed)

# 주성분별 기여도 확인
pca_weight = model.explainedVariance 
np.sum(pca_weight.toArray()) # 전체 기여도 합 확인

# 모든 데이터 PCA 변환
data_pca50 = model.transform(data_feet) 

# 결과 저장
data_pca50.write.parquet("PCA 결과 저장 경로")

# 저장된 결과 다시 불러오기
data_pca = spark.read.parquet("PCA 결과 저장 경로")

# seed 관측치의 주성분 벡터들의 평균 벡터값을 계산
from pyspark.ml.stat import Summarizer
pca_mean = data_pca.filter(F.col("seed")==1).select(Summarizer.mean(data_pca.pca_features).alias("pca_mean_vector"))
pca_mean_df = pca_mean.toPandas() # 값 사용 위해 pandas로 변환

# 가중치 부여한 유클리디안 거리 함수 설정
def euclidean_distance(a):
    b = pca_mean_df.pca_mean_vector[0].toArray()
    w = pca_weight.toArray()
    q = a-b
    return float(np.sqrt((w*q*q).sum()))
  
# 주성분 벡터들로 시드 관측치들의 평균 벡터와 각 관측치들의 벡터 간 거리 계산
from pyspark.sql.types import FloatType

data_pca_similar = data_pca.withColumn("euclidean", F.udf(euclidean_distance, FloatType())(F.col("pca_features")))

# 유사도 계산 데이터 저장
data_pca_similar.write.parquet("유사도 데이터 저장 경로")
# 유사도 계산 데이터 불러오기
data_pca_similar = spark.read.parquet("유사도 데이터 저장 경로")

