"""
Diagnostic : pourquoi le top-1 a raté un match évident (pare-feu)
=====================================================================

But : déterminer si le problème vient d'un BUG dans le calcul original
de recherche du meilleur candidat (top-1), ou d'une vraie limite du
modèle Sentence-BERT sur ce type de paraphrase technique française.

Si "Mise en place d'un pare-feu réseau" ressort avec un score ÉLEVÉ ici
mais qu'il n'a PAS été sélectionné comme meilleur candidat dans la
recherche top-1 originale → bug de calcul à corriger (re-exécuter
proprement la recherche du meilleur candidat suffira).

Si son score est BAS même recalculé ici → vraie limite du modèle,
à documenter comme limite dans l'article.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✅ Modèle chargé")

df_pe = pd.read_csv("/content/drive/MyDrive/alignement/PE_SN_segments_harmonized.csv")
segments_pe = df_pe['texte_source'].tolist()
modules_pe = df_pe['module_id'].tolist()

# =========================================================================
# TEST 1 : comparaison directe, un contre un (pas de boucle, aucune
# ambiguïté possible sur ce qui est comparé à quoi)
# =========================================================================

cas_test = [
    ("Mettre en place les règles de pare-feu", "Mise en place d'un pare-feu réseau"),
    ("Mettre en place les règles de pare-feu", "Sauvegarde en ligne"),  # l'ancien "meilleur"
    ("Configurer les adresses IP", "IPv4"),
    ("Configurer les adresses IP", "IPsec"),  # l'ancien "meilleur"
]

print("\n" + "=" * 70)
print("TEST 1 : COMPARAISON DIRECTE (paire par paire)")
print("=" * 70)
for texte1, texte2 in cas_test:
    emb1 = model.encode([texte1])
    emb2 = model.encode([texte2])
    sim = cosine_similarity(emb1, emb2)[0][0]
    print(f"\n  \"{texte1}\"")
    print(f"  vs \"{texte2}\"")
    print(f"  → score : {sim:.4f}")

# =========================================================================
# TEST 2 : classement complet — où se situe le segment "pare-feu"
# parmi TOUS les 312 segments pour cette compétence précise ?
# =========================================================================

def classement_complet(competence, segments_pe, modules_pe, model, top_n=10):
    emb_comp = model.encode([competence])
    resultats = []
    for seg, mod in zip(segments_pe, modules_pe):
        emb_seg = model.encode([seg])
        sim = cosine_similarity(emb_comp, emb_seg)[0][0]
        resultats.append((seg, mod, sim))
    resultats.sort(key=lambda x: x[2], reverse=True)
    return resultats[:top_n]

print("\n" + "=" * 70)
print("TEST 2 : TOP 10 COMPLET pour \"Mettre en place les règles de pare-feu\"")
print("=" * 70)
top10_parefeu = classement_complet(
    "Mettre en place les règles de pare-feu", segments_pe, modules_pe, model
)
for rang, (seg, mod, sim) in enumerate(top10_parefeu, 1):
    marqueur = " ⬅️ ANCIEN 'MEILLEUR' RETENU" if seg == "Sauvegarde en ligne" else ""
    marqueur2 = " ⬅️ CANDIDAT ÉVIDENT ATTENDU" if "pare-feu" in seg.lower() or "firewall" in seg.lower() else ""
    print(f"  {rang}. [{sim:.4f}] {seg} ({mod}){marqueur}{marqueur2}")

print("\n" + "=" * 70)
print("TEST 2bis : TOP 10 COMPLET pour \"Configurer les adresses IP\"")
print("=" * 70)
top10_ip = classement_complet(
    "Configurer les adresses IP", segments_pe, modules_pe, model
)
for rang, (seg, mod, sim) in enumerate(top10_ip, 1):
    marqueur = " ⬅️ ANCIEN 'MEILLEUR' RETENU" if seg == "IPsec" else ""
    marqueur2 = " ⬅️ CANDIDAT ATTENDU" if seg == "IPv4" else ""
    print(f"  {rang}. [{sim:.4f}] {seg} ({mod}){marqueur}{marqueur2}")

print("\n" + "=" * 70)
print("INTERPRÉTATION")
print("=" * 70)
print("""
Si "Mise en place d'un pare-feu réseau" apparaît en position #1 ou #2
du Test 2 avec un score élevé (>0.8) → la recherche top-1 originale
avait un bug (possiblement un problème de cache, d'index, ou
d'exécution partielle). Re-exécuter la recherche depuis zéro devrait
suffire à corriger ça.

S'il reste loin dans le classement même ici → c'est une vraie limite
du modèle sur cette paraphrase, à documenter dans l'article plutôt
qu'à essayer de corriger par du bricolage.
""")
