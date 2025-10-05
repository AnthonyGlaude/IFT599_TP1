"""
fichier : recommender.py
auteur.e : Anthony Glaude (glaa3301)
auteur.e : Étienne Chaput (chae3018)
auteur.e : Ana Karen Lopez Carbajal (lopa2603)
description : fichier permettant de faire une récommendation d'achats de clients
date : 10/05/2025
"""

from __future__ import annotations
from typing import List, Set
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# 1) Baskets (carte × fenêtre) 
def build_baskets_with_keys(
    df: pd.DataFrame,
    id_col: str = "card_id",
    time_col: str = "date",
    item_col: str = "mcc_code",
    window_seconds: int = 24*3600,
    min_items: int = 2,
    prefix: str = "mcc_",
):
    """
    Comme votre version: regroupe en sessions temporelles par carte
    et retourne une liste de paniers DISTINCTS avec la clé (carte, session).
    """
    if df.empty:
        return []
    # 1 Nettoyage / tri 
    df_clean = df.dropna(subset=[id_col, time_col, item_col]).copy()
    df_clean[time_col] = pd.to_datetime(df_clean[time_col], errors="coerce")
    df_clean = df_clean.dropna(subset=[time_col]).sort_values([id_col, time_col])
    # 2 Découpage en sessions 
    gap = df_clean.groupby(id_col)[time_col].diff().dt.total_seconds().fillna(np.inf)
    df_clean["_sess"] = (gap > float(window_seconds)).astype(int).groupby(df_clean[id_col]).cumsum()
    df_clean["_item"] = prefix + df_clean[item_col].astype(str)
    # 3 Agrégation 
    baskets = []
    for (cid, s), grp in df_clean.groupby([id_col, "_sess"]):
        items = sorted(set(grp["_item"].tolist()))
        if len(items) >= int(min_items):
            baskets.append({"card_id": cid, "session": int(s), "items": items})
    return baskets

# 2) Règles d’association — FP-Growth 
def mine_rules_fpgrowth(
    baskets: List[List[str]],
    min_support: float = 0.001,
    min_confidence: float = 0.10,
    max_len: int = 2
) -> pd.DataFrame:
    if not baskets:
        return pd.DataFrame(columns=["antecedents","consequents","support","confidence","lift","leverage"])
    
    te = TransactionEncoder()
    X = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)
    fis = fpgrowth(X, min_support=min_support, use_colnames=True, max_len=max_len)
    if fis.empty:
        return pd.DataFrame(columns=["antecedents","consequents","support","confidence","lift","leverage"])
    rules = association_rules(fis, metric="confidence", min_threshold=min_confidence)
    # garder seulement les conséquents de taille 1
    rules = rules[rules["consequents"].apply(lambda s: len(s) == 1)].copy()
    # tri simple
    return rules.sort_values(["confidence","lift"], ascending=False).reset_index(drop=True)

# 3) Reco minimale
class Recommender:
    def __init__(self, rules: pd.DataFrame):
        """
        rules : DataFrame avec colonnes 'antecedents','consequents','confidence','lift','support','leverage'
        Seuils simples à durcir/assouplir ici si besoin.
        """
        if rules is None or rules.empty:
            self.rules = pd.DataFrame(columns=["antecedents","consequents","confidence","lift","support","leverage"])
        else:
            self.rules = rules[
                (rules["lift"] >= 1.0) &
                (rules["confidence"] >= 0.10) &
                (rules["support"] >= 0.01)&
                (rules['leverage'] > 0.001)
            ].copy()

    def recommend(self, products: Set[str], n: int = 5) -> List[str]:
        """
        Retourne jusqu'à n items recommandés, déduits des règles dont
        les antécédents sont inclus dans `products`.
        products : items déjà présents/observés (ex. {'mcc_5411','mcc_5541'})
        n        : nombre max de recommandations
        """
        # Règles pertinentes : celles dont l'antécédent est un sous-ensemble de `products`
        is_relevant = self.rules["antecedents"].apply(lambda antecedent: antecedent.issubset(products))
        relevant_rules = self.rules[is_relevant]
        if relevant_rules.empty:
            return []
        # Ordonner les règles par qualité
        relevant_rules = relevant_rules.sort_values(["confidence", "lift"], ascending=False)
        already_suggested: Set[str] = set()
        recommendations: List[str] = []
        # Parcourir les conséquents (taille 1 par construction)
        for consequent in relevant_rules["consequents"]:
            candidate = next(iter(consequent))  # un seul élément dans le set
            if (candidate in products) or (candidate in already_suggested):
                continue
            already_suggested.add(candidate)
            recommendations.append(candidate)
            if len(recommendations) >= n:
                break
        return recommendations

