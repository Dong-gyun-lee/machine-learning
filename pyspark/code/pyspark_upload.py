import pandas as pd
import pyarrow.orc
from glob import glob
import findspark
import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
import pyspark.sql.functions as F

# 판다스로 불러오기
path = 'directory/path/*' # 리눅스 환경 directory의 분할된 orc 파일 모두 불러오기
files = glob(path) # 분할 저장된 orc 파일 불러오기
df = pd.concat(pd.read_orc(file) for file in files)

#spark dataframe 사용하여 데이터 불러오기
appName = "PySpark Example - Read and Write orc"
master = "local[n]"  #사용할 수 있는 코어(n) 만큼 부여 / master = "local[*]" max core 사용
path2 = 'directory_path/part-*' # 리눅스 환경 directory의 parquet 파일 모두 불러오기
findspark.init()
spark = SparkSession.builder\
.appName(appName).master(master).config("spark.driver.memory", "10g")\
.config("spark.executor.memory", "10g")\
.config("spark.ui.port","0000").getOrCreate()  #config 옵션으로 memory, port와 같은 spark 환경 조절
date2 = spark.read.orc(path2)


# 데이터 Row type으로 만들어서 dataframe 만들기
datas = [Row(name='Alice', age=5, height=80),
          Row(name='Blice', age=7, height=100),
          Row(name='Clice', age=9, height=120),
          Row(name='Aob', age=11, height=140),
          Row(name='Bob', age=13, height=160),
          Row(name='Cob', age=15, height=170),
          Row(name='Max', age=17, height=175),
          Row(name='Nax', age=19, height=180)]

sc = SparkContext()
spark = SparkSession(sc)

datas_df = sc.parallelize(datas).toDF()
datas_df2 = spark.createDataFrame(datas)
