"""
TP2 – Analyse des données (IFT599 / IFT799)

**Auteurs :**  
-  Ana Karen Lopez Carbajal (lopa2603)
-  Étienne Chaput (chae3018)
-  Anthony Glaude (glaa3301)

**Date de remise :** 25 novembre 2025  
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import time
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score, roc_auc_score,)

def plot_k_distance_on_ax(ax, X, title, k=5):
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)
    dist_k = np.sort(distances[:, k-1])

    ax.plot(dist_k)
    ax.set_title(f"{title}")
    ax.set_xlabel("Points triés")
    ax.set_ylabel(f"distance au {k}e plus proche voisin")
    return dist_k

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


def estimate_optimal_threshold(scores_val, y_val, n_thresholds=200):
    scores_val = np.asarray(scores_val)

    best_thr = None
    best_f1 = -1.0

    thresholds = np.linspace(scores_val.min(), scores_val.max(), n_thresholds)

    for thr in thresholds:
        preds = (scores_val > thr).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr, best_f1

def compute_metrics(scores_val, y_val, scores_test, y_test, name="Modele"):

    best_thr, best_f1 = estimate_optimal_threshold(scores_val, y_val)
    print(f"[{name}] threshold (val):", best_thr, " meilleur F1 (val):", best_f1)

    preds_test = (scores_test > best_thr).astype(int)

    acc  = accuracy_score(y_test, preds_test)
    prec = precision_score(y_test, preds_test)
    rec  = recall_score(y_test, preds_test)
    f1   = f1_score(y_test, preds_test)
    auc  = roc_auc_score(y_test, scores_test)

    print(f"{name} Results:")
    print("Acc :", acc)
    print("Prec:", prec)
    print("Rec :", rec)
    print("F1 :", f1)
    print("ROC-AUC  :", auc)

    return {
        "threshold": best_thr,
        "val_f1": best_f1,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "roc-auc": auc
    }
