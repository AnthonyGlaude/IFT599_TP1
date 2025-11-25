"""
TP2 – Analyse des données (IFT599 / IFT799)

**Auteurs :**  
-  Ana Karen Lopez Carbajal (lopa2603)
-  Étienne Chaput (chae3018)
-  Anthony Glaude (glaa3301)

**Date de remise :** 25 novembre 2025  
"""

from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import time
import torch
import torch.nn as nn

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


class AE(nn.Module):
    def __init__(self, in_features):
        super(AE, self).__init__()
        self.name = 'AE'
        self.enc = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.dec = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, in_features),
        )

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode
