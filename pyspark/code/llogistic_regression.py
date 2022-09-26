data_not_seed = data.filter(F.col("seed")==0)
data_seed = data.filter(F.col("seed")==1)


## negative samling

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
.config('spark.ui.port','4051')\
.config('spark.local.dir', 'directory ')\
.getOrCreate()


## Logistic Regression
data = spark.read.parquet("data 경로")

ct_all = data_not_seed.count() # seed가 아닌 데이터
ct_seed = data_seed.count()
ne_portion = (ct_all-ct_seed*5)/ct_all  # negative sample을 positive sample의 5배로 설정함 
ne_cut = data_not_seed.select(F.percentile_approx("euclidean", [ne_portion], 100).alias("cut_value")) 
ne_cut_df = ne_cut.toPandas()
cut_value = ne_cut_df.cut_value[0][0]
data_negative = data_not_seed.filter(F.col("euclidean")>=cut_value)



# cross-validation 위한 cv column 설정 - 여기선 사용하지 않은 column / 참고로 적어놓음

posi_df = data_seed.withColumn('cv', (F.rand(seed=123)*5).cast('integer'))
nega_df = data_negative.withColumn('cv', (F.rand(seed=123)*5).cast('integer'))
data_train = posi_df.union(nega_df)

# test data 생성
data_test = data_not_seed.filter((F.col("euclidean")<=cut_value)&(F.col("seed")==0))

## Modeling
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

lr_cv = LogisticRegression(labelCol = "seed", featuresCol="pca_features")

# cross validation setting
lrparamGrid = (ParamGridBuilder()\
               .addGrid(lr_cv.regParam, [0.01, 0.5, 1.0])\
               .addGrid(lr_cv.elasticNetParam, [0.0, 0.5, 1.0])\
               .addGrid(lr_cv.maxIter, [1, 10, 20])\
               .build())

# Evaluate model
lrevaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol = "seed")

# validation
lrcv = CrossValidator(estimator = lr_cv,estimatorParamMaps = lrparamGrid,evaluator = lrevaluator,numFolds = 5)

lrModel = lr_cv.fit(data_train)

test_result = lrModel.transform(data_test)

# probability 결과에서 seed audience일 확률(target 변수가 1일 확률)만 가져옴

from pyspark.ml.functions import vector_to_array
result = test_result.withColumn("prob", vector_to_array("probability"))\
.select(["oaid","last_update_date","seed"]+[F.col("prob")[i] for i in range(2)]).drop("probability","prob[0]")\
.withColumnRenamed("prob[1]","seed_prob")

# 결과 저장
result.write.parquet("")
