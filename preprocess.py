import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import umap

def load_hiseq_data(data_path, labels_path):
    """Charge et prépare les données Hi-Seq"""
    hiseq_X = pd.read_csv(data_path)
    hiseq_y = pd.read_csv(labels_path)
    
    hiseq_X = hiseq_X.drop('Unnamed: 0', axis=1)
    hiseq_y = hiseq_y.drop('Unnamed: 0', axis=1)
    
    hiseq_data = pd.concat([hiseq_X, hiseq_y], axis=1)
    return hiseq_X, hiseq_y, hiseq_data

def preprocess_hiseq(X, y):
    """Prétraitement des données Hi-Seq"""

    print(f"Missing values: {X.isna().sum().sum()}")
    print(f"Duplicates: {pd.concat([X, y], axis=1).duplicated().sum()}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

def reduce_dimension_umap(X, n_components=100, n_neighbors=15, min_dist=0.1):
    
    print(f"Reducing dimension from {X.shape[1]} to {n_components}D using UMAP...")
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
    X_reduced = reducer.fit_transform(X)
    
    return X_reduced, reducer

def reduce_dimension_acp(X, n_components=100):
    
    print(f"Reducing dimension from {X.shape[1]} to {n_components}D using ACP...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    return X_pca, pca