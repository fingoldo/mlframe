from typing import Any,Sequence,List,Optional,Callable
import numpy as np

def list_cluster_members(labels:Sequence,true_lables:Sequence)->None:
    for group in range(max(labels)+1):
        print([true_lables[i] for i in np.where(labels==group)[0]])    
        
def clusterize(X:Optional[Any]=None,true_labels:Optional[Sequence]=None,clusterizer:Optional[object]=None,dim_reducer:Optional[object]=None,
               show_plot:Optional[bool]=True,show_metrics:Optional[bool]=True,list_members:Optional[bool]=True,title:str=None):    
    
    from sklearn import metrics
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler


    if X is None:
        # Generate sample data
        centers = [[1, 1], [-1, -1], [1, -1]]
        X, true_labels = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,random_state=0)

    if dim_reducer:X= dim_reducer.fit_transform(X)
    X = StandardScaler().fit_transform(X)

    # #############################################################################
    # Compute DBSCAN
    if clusterizer is None: clusterizer=DBSCAN(eps=0.8, min_samples=3)
    db = clusterizer.fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    if show_metrics:
        if len(np.unique(labels))>1:print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
        if true_labels:
            print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels, labels))
            print("Completeness: %0.3f" % metrics.completeness_score(true_labels, labels))
            print("V-measure: %0.3f" % metrics.v_measure_score(true_labels, labels))
            print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(true_labels, labels))
            print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(true_labels, labels))
        

    # #############################################################################
    # Plot result
    if show_plot:
        import matplotlib.pyplot as plt

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)

        if title:
            plt.title(title)
        else:
            plt.title('Estimated number of clusters: %d' % n_clusters_)
        if true_labels:
            for i in range(len(true_labels)):
                plt.annotate(true_labels[i],(X[i,0],X[i,1]))
        plt.axis('off')
        plt.show()
        
        if true_labels:
            if list_members: list_cluster_members(labels,true_labels)
    
    return labels