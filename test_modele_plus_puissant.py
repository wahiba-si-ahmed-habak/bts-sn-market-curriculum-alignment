"""
Test avec un modèle Sentence-BERT plus puissant
=====================================================================

MiniLM-L12-v2 est compact et rapide, mais visiblement insuffisant pour
certaines paraphrases techniques françaises (score de 0.24 pour
"Mettre en place les règles de pare-feu" vs "Mise en place d'un
pare-feu réseau" — alors que c'est presque le même sens).

Ce script teste mpnet-base-v2, un modèle plus puissant de la même
famille (sentence-transformers), sur les mêmes cas problématiques,
afin de déterminer si la limite identifiée est résolue.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Chargement de paraphrase-multilingual-mpnet-base-v2...")
print("(plus volumineux que MiniLM, le téléchargement prend plus de temps)")
model_fort = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
print("✅ Modèle chargé")

cas_test = [
    ("Mettre en place les règles de pare-feu", "Mise en place d'un pare-feu réseau"),
    ("Mettre en place les règles de pare-feu", "Sauvegarde en ligne"),
    ("Configurer les adresses IP", "IPv4"),
    ("Configurer les adresses IP", "IPsec"),
    ("Gérer le service DNS interne", "Surveillance et dépannage du service DNS"),
]

print("\n" + "=" * 70)
print("COMPARAISON : MiniLM (ancien) vs mpnet (nouveau)")
print("=" * 70)

scores_minilm = {
    ("Mettre en place les règles de pare-feu", "Mise en place d'un pare-feu réseau"): 0.2358,
    ("Mettre en place les règles de pare-feu", "Sauvegarde en ligne"): 0.6098,
    ("Configurer les adresses IP", "IPv4"): 0.6526,
    ("Configurer les adresses IP", "IPsec"): 0.7415,
    ("Gérer le service DNS interne", "Surveillance et dépannage du service DNS"): 0.8254,
}

for texte1, texte2 in cas_test:
    emb1 = model_fort.encode([texte1])
    emb2 = model_fort.encode([texte2])
    sim_mpnet = cosine_similarity(emb1, emb2)[0][0]
    sim_minilm = scores_minilm[(texte1, texte2)]

    print(f"\n  \"{texte1}\"")
    print(f"  vs \"{texte2}\"")
    print(f"  MiniLM (ancien) : {sim_minilm:.4f}")
    print(f"  mpnet (nouveau) : {sim_mpnet:.4f}")
    diff = sim_mpnet - sim_minilm
    fleche = "📈 amélioration nette" if diff > 0.1 else ("📉 dégradation" if diff < -0.05 else "≈ stable")
    print(f"  Écart : {diff:+.4f} {fleche}")

print("\n" + "=" * 70)
print("DÉCISION")
print("=" * 70)
print("""
Si mpnet corrige nettement le cas pare-feu (score >0.6-0.7) sans casser
les autres cas → on bascule tout le pipeline sur ce modèle, on refait
la calibration ET le recalcul Groupe 1/2 avec lui. Plus fiable, et un
choix de modèle plus robuste, justifiable et documentable dans l'article.
""")
