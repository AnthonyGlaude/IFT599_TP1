import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import time
import matplotlib.pyplot as plt

def evaluate_clustering(X, labels_true, labels_pred):
    """Évalue les performances du clustering"""

    valid = labels_pred != -1
    if valid.sum() < 2:  
        return None
    
    metrics = {}
    
    # Métriques internes
    metrics['silhouette'] = silhouette_score(X[valid], labels_pred[valid])
    metrics['davies_bouldin'] = davies_bouldin_score(X[valid], labels_pred[valid])
    metrics['calinski_harabasz'] = calinski_harabasz_score(X[valid], labels_pred[valid])
    
    # Métriques externes
    metrics['ARI'] = adjusted_rand_score(labels_true[valid], labels_pred[valid])
    metrics['NMI'] = normalized_mutual_info_score(labels_true[valid], labels_pred[valid])
    
    return metrics

def visualize_clustering(X_2d, labels_true, labels_pred, title):
    """Visualise les résultats du clustering"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Vérité terrain
    scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_true.astype('category').cat.codes, 
                          cmap='viridis', s=10)
    ax1.set_title(f'{title} - Vérité terrain')
    plt.colorbar(scatter1, ax=ax1)
    
    # Clustering
    scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_pred, cmap='viridis', s=10)
    ax2.set_title(f'{title} - Clustering')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.show()