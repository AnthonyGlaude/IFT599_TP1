#!/usr/bin/env python
# coding: utf-8

# ## 1. Préparation des données
# 
# ### Dataset: Hi-Seq

# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


# # Import des modules personnalisés
from preprocess import load_hiseq_data, preprocess_hiseq, reduce_dimension_umap, reduce_dimension_acp
from models import run_kmeans, run_dbscan, run_spectral
from utils import plot_k_distance_on_ax, evaluate_clustering


# **Exploration et préparation initiale**
# 
# Le jeu de données Hi-Seq est composé de deux fichiers:
# - hiseq_data.csv (20 531 variables d’expression génique)
# - hiseq_labels.csv (type de tumeur associé à chaque échantillon)
# 
# Après chargement, les deux fichiers ont été concaténés et vérifiés:
# - Valeurs manquantes : aucune valeur manquante n’a été détectée.
# - Doublons : aucun échantillon dupliqué n’a été trouvé.
# - Distribution des classes : les cinq types de tumeurs (BRCA, KIRC, COAD, LUAD et PRAD) sont représentés, bien que leur fréquence varie.

# In[11]:


hiseq_X, hiseq_y, hiseq_data = load_hiseq_data("../data/hiseq_data.csv", "../data/hiseq_labels.csv")

hiseq_data.head()


# In[12]:


hiseq_X_scaled = preprocess_hiseq(hiseq_X, hiseq_y)


# **Normalisation**
# 
# Comme UMAP est sensible à l’échelle des variables, les 20 531 caractéristiques numériques ont été standardisées à l’aide de StandardScaler, afin que chaque variable ait une moyenne de 0 et un écart-type de 1.
# 
# **Réduction de dimension avec UMAP (pour la visualisation)**
# 
# Pour visualiser ce jeu de données très haute dimension, nous avons appliqué UMAP sur les variables normalisées.
# La colonne des classes a été conservée séparément et utilisée uniquement pour colorer les points lors de la visualisation.
# 
# Paramètres utilisés :
# - n_components = 2
# - n_neighbors = 15
# - min_dist = 0.1
# 
# Cette étape permet d’obtenir une représentation 2D facilitant l’observation de la structure des données et la séparation (ou non) entre les types de tumeurs.

# In[13]:


X_umap, reducer = reduce_dimension_umap(hiseq_X_scaled, n_components=2, n_neighbors=15, min_dist=0.1)

plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:,0], X_umap[:,1], c=hiseq_y["Class"].astype('category').cat.codes, s=10)
plt.title("UMAP Visualization of HiSeq Data")
plt.show()


# ## 2. Clustering
# 
# ### K-Means sur les données complètes vs. représentation UMAP à 100 dimensions vs ACP à 100 dimensions
# 
# Après le prétraitement et la normalisation du jeu de données Hi-Seq, on a réduit sa dimensionnalité d’origine à 100 attributs à l’aide de UMAP.
# Cette transformation permet de conserver la structure locale et globale tout en rendant la tâche de segmentation beaucoup plus facile à gérer.
# 
# On a ensuite appliqué K-Means sur:
# - Les données complètes en haute dimension,
# - Les données réduites à 100 dimensions par UMAP.
# 
# Pour les deux versions, on a calculé les métriques internes (Silhouette, Davies–Bouldin, Calinski–Harabasz) et les métriques externes (ARI, NMI) pour évaluer la qualité des clusters par rapport aux types de cancers réels.

# In[14]:


# Reduced dataset UMAP 100D
hiseq_X_UMAP_reduced, umap_model_100 = reduce_dimension_umap(hiseq_X_scaled, n_components=100, n_neighbors=15, min_dist=0.1)


# In[15]:


# Reduced dataset PCA 100D
hiseq_X_PCA_reduced, pca = reduce_dimension_acp(hiseq_X_scaled, n_components=100)


# In[16]:


labels_km_full, t_km_full = run_kmeans(hiseq_X_scaled, hiseq_y)
labels_km_UMAP_reduced, t_km_umap = run_kmeans(hiseq_X_UMAP_reduced, hiseq_y)
labels_km_PCA_reduced, t_km_umap = run_kmeans(hiseq_X_PCA_reduced, hiseq_y)


# In[17]:


hiseq_y = hiseq_y.squeeze()
results_km_full = evaluate_clustering(hiseq_X_scaled, hiseq_y, labels_km_full)
results_km_UMAP_reduced = evaluate_clustering(hiseq_X_UMAP_reduced, hiseq_y, labels_km_UMAP_reduced)
results_km_PCA_reduced = evaluate_clustering(hiseq_X_PCA_reduced, hiseq_y, labels_km_PCA_reduced)


# In[18]:


# Merge results in a dataframe
df_km = pd.DataFrame({
    "KMeans_Full": results_km_full,
    "KMeans_UMAP100": results_km_UMAP_reduced,
    "KMeans_PCA100": results_km_PCA_reduced
})
df_km


# ### Conclusions
# 
# Les résultats montrent une énorme amélioration lorsque la segmentation est faite sur les données réduites avec UMAP:
# 
# - Silhouette passe de 0.13 → 0.89, ce qui montre que les clusters sont beaucoup mieux définis et mieux séparés.
# - Davies–Bouldin chute de 2.64 → 0.15, indiquant des groupes beaucoup plus compacts.
# - Calinski–Harabasz saute de 65.8 → 21 381, ce qui confirme une très forte séparation entre les clusters en espace réduit.
# - ARI s’améliore de 0.79 → 0.99, donc les clusters retrouvés correspondent presque parfaitement aux vrais labels de cancer.
# - NMI augmente de 0.85 → 0.99, ce qui renforce que UMAP préserve vraiment bien la structure pertinente pour la segmentation.
# 
# Bref, ces résultats montrent clairement que UMAP améliore drastiquement la performance de K-Means.
# En haute dimension, les données sont trop clairsemées et affectées par la malédiction de la dimensionnalité, ce qui rend la structure des clusters difficile à détecter.
# Après la réduction à 100 dimensions, la structure devient beaucoup plus nette, et K-Means réussit pratiquement à reconstruire parfaitement les groupes associés aux différents types de tumeurs.

# In[19]:


get_ipython().system('jupyter nbconvert --to script main.ipynb')


# ### DBSCAN sur les données complètes vs. représentation UMAP à 100 dimensions

# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(18, 4)) 
# 1. HiSeq prétraité 
distances_k_full = plot_k_distance_on_ax(axes[0],hiseq_X_scaled,title="k-distances DBSCAN (HiSeq prétraité)",k=5)
# 2. UMAP 100D
distances_k_umap = plot_k_distance_on_ax(axes[1],hiseq_X_UMAP_reduced, title="k-distances DBSCAN (UMAP 100D)",k=5)
# 3. ACP 100D
distances_k_acp = plot_k_distance_on_ax(axes[2],hiseq_X_PCA_reduced, title="k-distances DBSCAN (UMAP 100D)",k=5)


# In[ ]:


# HiSeq prétraité
hiseq_y = hiseq_y.squeeze() # au cas ou 
labels_db_full, dur_full = run_dbscan(hiseq_X_scaled, hiseq_y, eps=190, min_samples=5)
metrics_db_full = evaluate_clustering(hiseq_X_scaled, labels_db_full, hiseq_y)

# UMAP 100D
labels_db_umap, dur_umap = run_dbscan(hiseq_X_UMAP_reduced, hiseq_y, eps=0.40, min_samples=5)
metrics_db_umap = evaluate_clustering(hiseq_X_UMAP_reduced, labels_db_umap, hiseq_y)

# ACP 100D
labels_db_acp, dur_umap = run_dbscan(hiseq_X_PCA_reduced, hiseq_y, eps=150, min_samples=5)
metrics_db_acp= evaluate_clustering(hiseq_X_PCA_reduced, labels_db_acp, hiseq_y)


# ### Conclusions
# 
# Les résultats montrent une amélioration massive lorsque la segmentation est effectuée sur les données réduites par UMAP (100 dimensions) plutôt que sur les données HiSeq prétraitées :
# 
# - La silhouette passe d’environ 0.14 à 0.89, montrant des clusters beaucoup mieux définis et nettement séparés.
# 
# - L’indice de Davies–Bouldin chute d’environ 2.6 à 0.14, ce qui indique des groupes beaucoup plus compacts et bien séparés.
# 
# - L’indice de Calinski–Harabasz augmente d’environ 65 à 16 769, signe d’une structure de clusters très marquée dans l’espace UMAP.
# 
# - Les indices externes s’améliorent également de façon spectaculaire : l’ARI passe d’environ 0 à 0.99 et la NMI d’environ 0 à 0.98, ce qui signifie que les clusters retrouvés correspondent presque parfaitement aux classes réelles de cancer.
# 
# Ainsi, l'application de l'algorithme de clustering DSCAN directement sur les données HiSeq prétraitées ne parvient pas à exploiter la structure de classe. En vérifiant le nombre de clusters ainsi que les bruits, nous observons qu’il ne détecte qu’un seul cluster avec 22 clusters de bruits(1). Cela confirme son incapacité à séparer correctement les classes dans l'espace.
# 
# À l’inverse, après réduction de dimension par UMAP, DBSCAN identifie cinq clusters bien séparés (n_clusters = 5, n_noise = 2) qui correspondent presque parfaitement aux étiquettes réelles. Cela montre que la réduction de dimension non linéaire par UMAP rend la structure des données beaucoup plus exploitable pour un algorithme de clustering basé sur la densité comme DBSCAN.
# 
# (1) Documentation DBSCAN de scikit-learn
# 

# In[ ]:


df_dbscan = pd.DataFrame({
    "DBSCAN_Full":   metrics_db_full,
    "DBSCAN_UMAP100": metrics_db_umap,
    "DBSCAN_PCA100": metrics_db_acp
})
df_dbscan
df_dbscan
def cluster_stats(labels):
    n_noise = np.sum(labels == -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters, n_noise

n_clusters_full, n_noise_full = cluster_stats(labels_db_full)
n_clusters_umap, n_noise_umap = cluster_stats(labels_db_umap)
n_clusters_acp, n_noise_acp = cluster_stats(labels_db_acp)

print("Full  : n_clusters =", n_clusters_full, ", n_noise =", n_noise_full)
print("UMAP  : n_clusters =", n_clusters_umap, ", n_noise =", n_noise_umap)
print("ACP  : n_clusters =", n_clusters_acp, ", n_noise =", n_noise_acp)
print(df_dbscan)


# ### 3. Spectral Clustering sur les données complètes vs. représentation UMAP à 100 dimensions
# On applique maintenant Spectral Clustering sur:
# - Les données complètes en haute dimension,
# - Les données réduites à 100 dimensions par UMAP.
# 
# Même évaluation avec les métriques internes et externes.

# In[21]:


print("SPECTRAL CLUSTERING - EXÉCUTION")
print("=" * 50)

print("1. Spectral Clustering sur données complètes...")
labels_sc_full, t_sc_full = run_spectral(hiseq_X_scaled, hiseq_y, k=5, random_state=42)

print("2. Spectral Clustering sur données réduites UMAP...")
labels_sc_reduced, t_sc_reduced = run_spectral(hiseq_X_UMAP_reduced, hiseq_y, k=5, random_state=42)

print("3. Évaluation des performances...")
results_sc_full = evaluate_clustering(hiseq_X_scaled, hiseq_y, labels_sc_full)
results_sc_reduced = evaluate_clustering(hiseq_X_UMAP_reduced, hiseq_y, labels_sc_reduced)

df_sc = pd.DataFrame({
    "Spectral_Full": results_sc_full,
    "Spectral_UMAP100": results_sc_reduced
})

print("TERMINÉ - Résultats du Spectral Clustering")
df_sc


# In[22]:


print("COMPARAISON K-MEANS vs SPECTRAL CLUSTERING")
print("=" * 60)

comparison_data = {}

# K-Means results
for metric in df_km.columns:
    comparison_data[metric] = df_km[metric]

# Spectral Clustering results  
for metric in df_sc.columns:
    comparison_data[metric] = df_sc[metric]

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison)

