from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

## 실루엣 계수 확인
silhouette_scores=[]
centers = []
evaluator = ClusteringEvaluator(featuresCol='features', metricName='silhouette', distanceMeasure='squaredEuclidean')

start = time.time()
for K in range(2,6):
    print(K)
    KMeans_=KMeans(featuresCol='features', k=K)
    KMeans_fit=KMeans_.fit(data)
    KMeans_transform=KMeans_fit.transform(data) 
    center = Kmeans_fit.clusterCenters()
    evaluation_score=evaluator.evaluate(KMeans_transform)
    silhouette_scores.append(evaluation_score)
    centers.append(center)
    KMeans_fit.setPredictionCol("cluster_"+str(K))
    data = model.transform(data)
    
end = time.time()
print(f"time:{end-start} sec")

## 실루엣계수를 보고 중심갯수 설정하여 클러스터링 결과 확인

k_number = np.argmax(silhouette_scores)+2 # 시작이 2부터라서 2 더해줌
data_cluster = data.select("cluster"+str(k_number))
data_center = centers[k_number-2]
