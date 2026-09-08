# -*- coding: utf-8 -*-
"""
Baseline Jaccard : alignement par similarité lexicale
=========================================================================

Calcule la similarité de Jaccard (intersection/union des mots) entre
chaque compétence extraite du Groupe 2 et les 312 segments du
référentiel PE, sans aucun modèle d'embeddings. Sert de borne
inférieure de performance dans le Tableau 7 de l'article.

Chiffres produits par ce script, cités dans l'article (Section 4.6) :
  - DHCP → DHCP : Jaccard = 1.00 (correspondance exacte)
  - "configurer un routeur" → "Configuration du routage réseau" :
    Jaccard = 0.00 (échec total sur la paraphrase)
  - Similarité moyenne sur le Groupe 2 : ≈ 0.28
"""

import pandas as pd
import numpy as np

# =========================================================================
# CHARGEMENT
# =========================================================================

df_entites = pd.read_csv("/content/drive/MyDrive/alignement/entites_test_gaps.csv")
competences = df_entites['texte_entite'].unique().tolist()

df_pe = pd.read_csv("/content/drive/MyDrive/alignement/PE_SN_segments_harmonized.csv")
segments_pe = df_pe['texte_source'].tolist()
modules_pe = df_pe['module_id'].tolist()

print(f"Compétences à aligner : {len(competences)}")
print(f"Segments PE : {len(segments_pe)}")


# =========================================================================
# FONCTION JACCARD
# =========================================================================

def jaccard_similarity(texte1, texte2):
    """Similarité de Jaccard entre deux textes (mots en commun / union)."""
    mots1 = set(texte1.lower().split())
    mots2 = set(texte2.lower().split())
    intersection = mots1 & mots2
    union = mots1 | mots2
    if len(union) == 0:
        return 0.0
    return len(intersection) / len(union)


# =========================================================================
# ALIGNEMENT PAR JACCARD (meilleur candidat parmi les 312 segments)
# =========================================================================

resultats_jaccard = []
for comp in competences:
    best_sim, best_seg, best_mod = 0, "", ""
    for seg, mod in zip(segments_pe, modules_pe):
        sim = jaccard_similarity(comp, seg)
        if sim > best_sim:
            best_sim, best_seg, best_mod = sim, seg, mod
    resultats_jaccard.append({
        'competence': comp,
        'jaccard_max': round(best_sim, 4),
        'meilleur_match': best_seg,
        'module': best_mod
    })

df_jaccard = pd.DataFrame(resultats_jaccard)

print("\n" + "=" * 70)
print("RÉSULTATS BASELINE JACCARD")
print("=" * 70)
print(f"   Similarité Jaccard moyenne : {df_jaccard['jaccard_max'].mean():.4f}")
print(f"   Similarité Jaccard max     : {df_jaccard['jaccard_max'].max():.4f}")
print(f"   Similarité Jaccard min     : {df_jaccard['jaccard_max'].min():.4f}")

df_jaccard.to_csv("/content/drive/MyDrive/alignement/baseline_jaccard.csv", index=False)
print(f"\nSauvegardé : baseline_jaccard.csv")


# =========================================================================
# ILLUSTRATION : Jaccard face à une paraphrase vs un match exact
# (cas cités explicitement dans l'article, Section 4.6)
# =========================================================================

cas_test = [
    ("configurer un routeur", "Configuration du routage réseau",
     "Paraphrase (aucun mot commun malgré le même sens)"),
    ("DHCP", "DHCP",
     "Correspondance exacte"),
    ("Déployer des conteneurs Docker", "Création des VLAN",
     "Faux positif possible (mot commun \"des\", sens différent)"),
]

print("\n" + "=" * 70)
print("ILLUSTRATION : LIMITES DE JACCARD")
print("=" * 70)
for comp, seg, explication in cas_test:
    jac = jaccard_similarity(comp, seg)
    print(f"\n  \"{comp}\" vs \"{seg}\"")
    print(f"  Jaccard = {jac:.2f} — {explication}")
