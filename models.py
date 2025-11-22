from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import time

def run_kmeans(X, y, k=5, random_state=42):
    """Exécute K-Means avec chronométrage"""
    start = time.time()
    km = KMeans(n_clusters=k, random_state=random_state)
    labels_pred = km.fit_predict(X)
    duration = time.time() - start
    return labels_pred, duration

def run_dbscan(X, y, eps=0.5, min_samples=5):
    """Exécute DBSCAN avec chronométrage"""
    start = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels_pred = dbscan.fit_predict(X)
    duration = time.time() - start
    return labels_pred, duration

def run_spectral(X, y, k=5, random_state=42):
    """Exécute Spectral Clustering avec chronométrage"""
    start = time.time()
    spectral = SpectralClustering(n_clusters=k, random_state=random_state, 
                                affinity='rbf', n_neighbors=10)
    labels_pred = spectral.fit_predict(X)
    duration = time.time() - start
    return labels_pred, duration