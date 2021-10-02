from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random
import numpy as np

def createClusteredData(N,k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    for i in range(k):
        incomeCentroid = random.uniform(20000.0,20000.0)
        ageCentroid = random.uniform(20.0,70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid,10000.0),np.random.normal(ageCentroid,2.0)])
        X = np.array(X)
        return X
        

data = createClusteredData(5000,5)
model = KMeans(n_clusters=5)
model = model.fit(scale(data)) #scaling for good results

print(model.labels_)
plt.figure(figsize= (8,6))
plt.scatter(data[:,0],data[:,1],c = model.labels_.astype(np.float))
print(plt.show())