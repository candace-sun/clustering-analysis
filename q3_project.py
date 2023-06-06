from mlxtend.data import iris_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pyclustering
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import random_center_initializer, kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans_visualizer
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.silhouette import silhouette
from sklearn.metrics import silhouette_score 
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    X, y = iris_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1, test_size=0.3)
    
    sc_X = StandardScaler()
    X_train=sc_X.fit_transform(X_train)
    X_test=sc_X.transform(X_test)
    
    km = KMeans(n_clusters=3,init="k-means++",random_state=2).fit(X)
    y_kmeans = km.predict(X)
    
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],s = 100, c = 'red', label = 'Iris-setosa')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],s = 100, c = 'blue', label = 'Iris-versicolour')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],s = 100, c = 'green', label = 'Iris-virginica')   #Visualising the clusters - On the first two columns
    plt.scatter(km.cluster_centers_[:,   0], km.cluster_centers_[:,1],s = 100, c = 'black', label = 'Centroids')   #plotting the centroids of the clusters
    plt.legend()
    plt.show()
    
    correct = 0
    for r in range(len(y_kmeans)):
        if y_kmeans[r] == y[r]:
            correct += 1

    manhattan_metric = distance_metric(type_metric.MANHATTAN)
    initial_centers = random_center_initializer(X, 3).initialize()
    kmeans_instance = kmeans(X, initial_centers, metric=manhattan_metric,random_state=3)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    #kmeans_visualizer.show_clusters(X,clusters,kmeans_instance.get_centers())
    #visualizer = cluster_visualizer_multidim()
    #visualizer.append_clusters(clusters)
    #visualizer.show(max_row_size=3)
    
    
    #newX = PCA(n_components=2).fit_transform(X)
    labels = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='single').fit_predict(X)
    hier_eu_correct = 0
    for r in range(len(y)):
        if labels[r] == y[r]:
            hier_eu_correct += 1
    
    mlabels = AgglomerativeClustering(n_clusters=5, affinity='manhattan', linkage='complete').fit_predict(X)
    hier_m_correct = 0
    for r in range(len(y)):
        if mlabels[r] == y[r]:
            hier_m_correct += 1
    
    i = 0
    m_correct = 0
    mkmeans = [ 0 ] * 150
    print(clusters)
    for cluster in clusters:
        for x in cluster:
            if y[x] == i:
                m_correct += 1
                mkmeans[x] = i
        i += 1
    
    
    clustering = DBSCAN(eps=0.5, min_samples=11).fit(X)
    print(clustering.labels_)
    d_correct = 0
    for r in range(len(y)):
        if clustering.labels_[r] == y[r]:
            d_correct += 1
            
    f_clustering = DBSCAN(eps=0.5, min_samples=11,algorithm='kd_tree').fit(X)
    print(f_clustering.labels_)
    f_correct = 0
    for r in range(len(y)):
        if f_clustering.labels_[r] == y[r]:
            f_correct += 1
    
    print(labels)
    print(confusion_matrix(y, clustering.labels_))
    ConfusionMatrixDisplay.from_predictions(y, clustering.labels_)

    plt.show()
    
    print("PERFORMANCE ACCURACY\n")
    
    print("1. K-means clustering algorithms")
    print("\tEuclidean dist clustering:",correct/len(y))
    print("\tManhattan dist clustering:",m_correct/len(y))
    print("\tDensity based clustering:",d_correct/len(y))
    print("\tFilter clustering:",f_correct/len(y))
    
    print("2. Hierarchical clustering algorithms")
    print("\tEuclidean dist clustering:",hier_eu_correct/len(y))
    print("\tManhattan dist clustering:",hier_m_correct/len(y))
      
if __name__ == "__main__":
    main()