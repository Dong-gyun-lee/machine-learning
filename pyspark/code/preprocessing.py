import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
rom functools import reduce
from pyspark.ml.stat import Summarizer
from pyspark.ml.functions import vector_to_array


## 1. 여러 column concat
data=data.select(F.concat_ws(" ",F.col("column_1"),F.col("column_2"),F.col("column_1")).alias("column_concat"))

## 2. column_name의 값 기준으로 여러 row의 문자열 데이터 한 column에 join , udf function 이용함

def sorter(cate_list):
  return '/'.join([item for item in cate_list]) #항목만 뽑아서 / 로 조인
sort_udf = F.udf(sorter,StringType())

data= data.agg(F.collect_list(F.col("column_name")).alias("text_list")).withColumn("texts",sort_udf("text_list")).drop("text_list")

## 3. vector type column의 mean vector 구하기

data = data.agg(Summarizer.mean(F.col("features")).alias("mean_features"))

## 4. 한 column의 vector값들을 여러 column으로 분할

data = (data.withColumn("vector_column", vector_to_array("array_column"))).select([F.col("vector_column")[i] for i in range(vetor column의 벡터 길이 값)])
oldColumns = data.schema.names
newColumns = ["new_column_lists"]
avg_kmeans = reduce(lambda data, idx: data.withColumnRenamed(oldColumns[idx], newColumns[idx]),range(len(oldColumns)), data)


## 5. column to row 방법 (pandas의 melt? 같은 효과)
from pyspark.sql.functions import array, col, explode, struct, lit

def to_long(df):
  # Filter dtypes and split into column names and type description
  cols, dtypes = zip(*((c, t) for (c, t) in df.dtypes))
  # Spark SQL supports only homogeneous columns
  assert len(set(dtypes)) == 1, "All columns have to be of the same type"
  # Create and explode an array of (column_name, column_value) structs
  kvs = explode(array([struct(lit(c).alias("column_name"), col(c).alias("new_column_name")) for c in cols])).alias("kvs")
  return df.select([kvs]).select(["kvs.column_name", "kvs.new_column_name"])

## 6. struct 구조 이용하여 groupBy
data_struct = data.groupBy(["cluster"]).agg(F.collect_list(struct("rank","category","value")).alias("struct_list"))

## 7. column별 rank 구하기
for colname in ["column_list"]:
    windowSpec  = Window.partitionBy("partition_column").orderBy(F.col(colname).desc())
    data = data.withColumn("rank_"+colname, rank().over(windowSpec))
