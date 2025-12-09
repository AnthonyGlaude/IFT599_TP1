# TP3 – Exploration du Web (IFT599 / IFT799)

**Auteurs :**  
-  Ana Karen Lopez Carbajal (lopa2603)
-  Étienne Chaput (chae3018)
-  Anthony Glaude (glaa3301)

**Date de remise :** 9 décembre 2025  

import time
import json
from app import crawl_and_build_graph, compute_pagerank, compute_hits

test_cases = [
    {
        "name": "Science des Données",
        "query": "science des données data science",
        "seeds": [
            "https://fr.wikipedia.org/wiki/Science_des_donn%C3%A9es",
            "https://www.kaggle.com/",
            "https://towardsdatascience.com/",
            "https://www.datacamp.com/",
            "https://medium.com/tag/data-science"
        ]
    },
    {
        "name": "Cybersécurité",
        "query": "cybersécurité sécurité informatique",
        "seeds": [
            "https://fr.wikipedia.org/wiki/Cybers%C3%A9curit%C3%A9",
            "https://www.cisa.gov/",
            "https://www.sans.org/",
            "https://owasp.org/",
            "https://krebsonsecurity.com/"
        ]
    },
    {
        "name": "Développement Web",
        "query": "développement web frontend backend",
        "seeds": [
            "https://developer.mozilla.org/fr/",
            "https://www.w3schools.com/",
            "https://stackoverflow.com/",
            "https://github.com/",
            "https://css-tricks.com/"
        ]
    },
    {
        "name": "Cloud Computing",
        "query": "cloud computing aws azure",
        "seeds": [
            "https://fr.wikipedia.org/wiki/Cloud_computing",
            "https://aws.amazon.com/fr/",
            "https://azure.microsoft.com/fr-fr/",
            "https://cloud.google.com/",
            "https://www.redhat.com/en/topics/cloud"
        ]
    },
    {
        "name": "Blockchain",
        "query": "blockchain cryptocurrency",
        "seeds": [
            "https://fr.wikipedia.org/wiki/Blockchain",
            "https://ethereum.org/",
            "https://bitcoin.org/",
            "https://www.coindesk.com/",
            "https://www.blockchain.com/"
        ]
    }
]

def run_manual_test(test_case):
    """Test manuel qui inclut le crawl"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_case['name']}")
    print(f"Query: {test_case['query']}")
    print(f"Seeds: {len(test_case['seeds'])}")
    print('='*60)
    
    print("Étape 1: Crawling des pages...")
    crawl_start = time.time()
    G = crawl_and_build_graph(test_case['seeds'])
    crawl_time = time.time() - crawl_start
    print(f"  - Pages trouvées: {G.number_of_nodes()}")
    print(f"  - Liens trouvés: {G.number_of_edges()}")
    print(f"  - Temps de crawl: {crawl_time:.2f}s")
    
    if G.number_of_nodes() == 0:
        print(" Aucune page crawlée!")
        return None
    
    print("\nÉtape 2: Calcul PageRank...")
    pr_start = time.time()
    pr_scores = compute_pagerank(G)
    pr_time = time.time() - pr_start
    top_pr = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"  - Temps PageRank: {pr_time:.3f}s")
    print(f"  - Top 3 PageRank:")
    for i, (url, score) in enumerate(top_pr[:3], 1):
        domain = url.split('/')[2] if len(url.split('/')) > 2 else url
        print(f"     {i}. {domain[:40]}... - Score: {score:.6f}")
    
    print("\nÉtape 3: Calcul HITS...")
    hits_start = time.time()
    hub_scores, auth_scores = compute_hits(G)
    hits_time = time.time() - hits_start
    top_auth = sorted(auth_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"  - Temps HITS: {hits_time:.3f}s")
    print(f"  - Top 3 HITS:")
    for i, (url, score) in enumerate(top_auth[:3], 1):
        domain = url.split('/')[2] if len(url.split('/')) > 2 else url
        print(f"     {i}. {domain[:40]}... - Score: {score:.6f}")
    
    print("\n" + "="*60)
    print("ANALYSE COMPARATIVE")
    print("="*60)
    
    pr_urls = {url for url, _ in top_pr}
    hits_urls = {url for url, _ in top_auth}
    common = pr_urls.intersection(hits_urls)
    
    print(f"Pages communes top 10: {len(common)}")
    print(f"Différence PageRank vs HITS: {len(pr_urls - hits_urls)} pages différentes")
    
    print(f"\nTemps de calcul (sans crawl):")
    print(f"  PageRank: {pr_time:.3f}s")
    print(f"  HITS: {hits_time:.3f}s")
    print(f"  Différence: {hits_time - pr_time:.3f}s")
    
    results = {
        'test_name': test_case['name'],
        'query': test_case['query'],
        'crawl_time': crawl_time,
        'graph_size': G.number_of_nodes(),
        'pagerank': {
            'time': pr_time,
            'top10': top_pr
        },
        'hits': {
            'time': hits_time,
            'top10': top_auth
        },
        'comparison': {
            'common_pages': list(common),
            'num_common': len(common)
        }
    }
    
    with open(f"results_{test_case['name'].replace(' ', '_')}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRésultats sauvegardés dans: results_{test_case['name'].replace(' ', '_')}.json")
    
    return results

def main():
    print("TEST MANUEL COMPLET - PageRank vs HITS")
    print("Ce script mesure le temps réel incluant le crawl")
    print("="*60)
    
    all_results = []
    
    for test in test_cases:
        result = run_manual_test(test)
        if result:
            all_results.append(result)
    
    print("\n" + "="*60)
    print("RÉCAPITULATIF GLOBAL DES 5 TESTS")
    print("="*60)
    
    if all_results:
        avg_pr_time = sum(r['pagerank']['time'] for r in all_results) / len(all_results)
        avg_hits_time = sum(r['hits']['time'] for r in all_results) / len(all_results)
        avg_crawl_time = sum(r['crawl_time'] for r in all_results) / len(all_results)
        avg_graph_size = sum(r['graph_size'] for r in all_results) / len(all_results)
        
        print(f"Temps moyen par test:")
        print(f"  - Crawl: {avg_crawl_time:.2f}s")
        print(f"  - PageRank: {avg_pr_time:.3f}s")
        print(f"  - HITS: {avg_hits_time:.3f}s")
        print(f"  - Total (crawl+algo): {avg_crawl_time + (avg_pr_time+avg_hits_time)/2:.2f}s")
        
        print(f"\nTaille moyenne du graphe: {avg_graph_size:.1f} pages")
        
        print(f"\nComparaison algorithmes:")
        print(f"  PageRank vs HITS: {avg_pr_time:.3f}s vs {avg_hits_time:.3f}s")
        print(f"  Différence: {avg_hits_time - avg_pr_time:.3f}s")
        print(f"  HITS est {((avg_hits_time/avg_pr_time)-1)*100:.1f}% plus {'rapide' if avg_hits_time < avg_pr_time else 'lent'}")
        
        avg_common = sum(r['comparison']['num_common'] for r in all_results) / len(all_results)
        print(f"\nPages communes moyenne top 10: {avg_common:.1f}")

if __name__ == "__main__":
    main()
