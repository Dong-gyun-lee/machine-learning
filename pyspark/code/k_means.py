from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

## 실루엣 계수 확인
silhouette_scores=[]
evaluator = ClusteringEvaluator(featuresCol='features', metricName='silhouette', distanceMeasure='squaredEuclidean')

start = time.time()
for K in range(2,6):
    
    print(K)
    KMeans_=KMeans(featuresCol='features', k=K)
    KMeans_fit=KMeans_.fit(data)
    KMeans_transform=KMeans_fit.transform(data) 
    evaluation_score=evaluator.evaluate(KMeans_transform)
    silhouette_scores.append(evaluation_score)
    
end = time.time()
print(f"time:{end-start} sec")

## 실루엣계수를 보고 중심갯수 설정하여 클러스터링
start = time.time()

k_number = np.argmax(silhouette_scores)+2
kmeans =KMeans(featuresCol='features', k=k_number)
kmeans_fit=kmeans.fit(data)
centers = kmeans_fit.clusterCenters()

end=time.time()
print(f"time:{end-start} sec")
