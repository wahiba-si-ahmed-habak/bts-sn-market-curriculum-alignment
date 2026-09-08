"""
Recalcul de l'alignement Groupe 1 et Groupe 2 avec le nouveau seuil
=======================================================================

Bonne nouvelle : pas besoin de refaire tourner Sentence-BERT sur toutes
les paires. Vos deux notebooks ont déjà sauvegardé, pour chaque
compétence, le MEILLEUR score de similarité trouvé (colonne
similarite_max / similarite). Il suffit de reclasser ALIGNE/GAP avec
le nouveau seuil à partir de ces scores déjà calculés — instantané.

Fichiers utilisés (déjà générés par les scripts d'alignement Groupe 1
et Groupe 2) :
  - resultats_alignement_corrige.csv   (Groupe 1, 8 offres, 106 compétences)
  - resultats_test_gaps.csv            (Groupe 2, 2 offres test, 51 compétences)
"""

import pandas as pd

NOUVEAU_SEUIL = 0.8257


def lire_csv_robuste(chemin):
    for encoding in ['utf-8', 'cp1252', 'latin-1']:
        for sep in [',', ';']:
            try:
                df = pd.read_csv(chemin, encoding=encoding, sep=sep)
                if len(df.columns) > 1:
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
    raise ValueError(f"Impossible de lire {chemin}")


# =========================================================================
# GROUPE 1 (8 offres, 106 compétences)
# =========================================================================

print("=" * 70)
print(f"🔍 GROUPE 1 — RECLASSIFICATION AU NOUVEAU SEUIL ({NOUVEAU_SEUIL})")
print("=" * 70)

df_g1 = lire_csv_robuste(
    "/content/drive/MyDrive/alignement/resultats_alignement_corrige.csv"
)

ancien_alignes_g1 = (df_g1['statut'] == 'ALIGNE').sum()
ancien_gaps_g1 = (df_g1['statut'] == 'GAP').sum()

df_g1['statut_nouveau'] = df_g1['similarite_max'].apply(
    lambda s: 'ALIGNE' if s >= NOUVEAU_SEUIL else 'GAP'
)

nouveau_alignes_g1 = (df_g1['statut_nouveau'] == 'ALIGNE').sum()
nouveau_gaps_g1 = (df_g1['statut_nouveau'] == 'GAP').sum()
total_g1 = len(df_g1)

print(f"\n   Ancien seuil (0.55)   : {ancien_alignes_g1} ALIGNÉS "
      f"({ancien_alignes_g1/total_g1*100:.1f}%) / {ancien_gaps_g1} GAPS")
print(f"   Nouveau seuil ({NOUVEAU_SEUIL}) : {nouveau_alignes_g1} ALIGNÉS "
      f"({nouveau_alignes_g1/total_g1*100:.1f}%) / {nouveau_gaps_g1} GAPS")

# Compétences qui basculent de ALIGNE -> GAP avec le nouveau seuil
bascules_g1 = df_g1[(df_g1['statut'] == 'ALIGNE') & (df_g1['statut_nouveau'] == 'GAP')]
print(f"\n   ⚠️ {len(bascules_g1)} compétences basculent ALIGNÉ → GAP :")
for _, row in bascules_g1.sort_values('similarite_max', ascending=False).iterrows():
    print(f"      • \"{row['competence_marche']}\" (sim: {row['similarite_max']:.3f}) "
          f"→ {row['module_pe']}")

df_g1.to_csv(
    "/content/drive/MyDrive/alignement/resultats_alignement_seuil_final.csv",
    index=False, encoding='utf-8'
)

# =========================================================================
# GROUPE 2 (2 offres test, 51 compétences, avec 7 GAPS attendus)
# =========================================================================

print("\n" + "=" * 70)
print(f"🔍 GROUPE 2 — RECLASSIFICATION AU NOUVEAU SEUIL ({NOUVEAU_SEUIL})")
print("=" * 70)

df_g2 = lire_csv_robuste(
    "/content/drive/MyDrive/alignement/resultats_test_gaps.csv"
)

ancien_alignes_g2 = (df_g2['statut'] == 'ALIGNE').sum()
ancien_gaps_g2 = (df_g2['statut'] == 'GAP').sum()

df_g2['statut_nouveau'] = df_g2['similarite'].apply(
    lambda s: 'ALIGNE' if s >= NOUVEAU_SEUIL else 'GAP'
)

nouveau_alignes_g2 = (df_g2['statut_nouveau'] == 'ALIGNE').sum()
nouveau_gaps_g2 = (df_g2['statut_nouveau'] == 'GAP').sum()
total_g2 = len(df_g2)

print(f"\n   Ancien seuil (0.55)   : {ancien_alignes_g2} ALIGNÉS "
      f"({ancien_alignes_g2/total_g2*100:.1f}%) / {ancien_gaps_g2} GAPS")
print(f"   Nouveau seuil ({NOUVEAU_SEUIL}) : {nouveau_alignes_g2} ALIGNÉS "
      f"({nouveau_alignes_g2/total_g2*100:.1f}%) / {nouveau_gaps_g2} GAPS")

# Vérification sur les 7 GAPS attendus (docker, python, zabbix, cloud,
# nessus, siem, stormshield) — LE test de robustesse le plus important
print(f"\n   🔍 VÉRIFICATION SUR LES 7 GAPS ATTENDUS (nouveau seuil) :")
attendus_gap = ['docker', 'python', 'zabbix', 'cloud', 'nessus', 'siem', 'stormshield']
df_ok_nouveau = df_g2[df_g2['statut_nouveau'] == 'ALIGNE']
df_gap_nouveau = df_g2[df_g2['statut_nouveau'] == 'GAP']
n_corrects = 0
for terme in attendus_gap:
    dans_gap = any(terme in c.lower() for c in df_gap_nouveau['competence'])
    dans_ok = any(terme in c.lower() for c in df_ok_nouveau['competence'])
    if dans_gap:
        print(f"      ✅ \"{terme}\" → GAP (correct)")
        n_corrects += 1
    elif dans_ok:
        print(f"      ⚠️ \"{terme}\" → ALIGNÉ (toujours un problème ?)")
    else:
        print(f"      ❓ \"{terme}\" → pas extrait")
print(f"\n   Score de détection : {n_corrects}/7 "
      f"({n_corrects/7*100:.1f}%) — ancien seuil donnait 5/7 (71.4%)")

df_g2.to_csv(
    "/content/drive/MyDrive/alignement/resultats_test_gaps_seuil_final.csv",
    index=False, encoding='utf-8'
)

# =========================================================================
# RÉSUMÉ POUR L'ARTICLE
# =========================================================================

print("\n" + "=" * 70)
print("📋 RÉSUMÉ POUR L'ARTICLE")
print("=" * 70)
print(f"""
Groupe 1 : {nouveau_alignes_g1}/{total_g1} alignés ({nouveau_alignes_g1/total_g1*100:.1f}%)
           [ancien seuil 0.55 : {ancien_alignes_g1}/{total_g1} = {ancien_alignes_g1/total_g1*100:.1f}%]

Groupe 2 : {nouveau_alignes_g2}/{total_g2} alignés ({nouveau_alignes_g2/total_g2*100:.1f}%)
           [ancien seuil 0.55 : {ancien_alignes_g2}/{total_g2} = {ancien_alignes_g2/total_g2*100:.1f}%]

Détection GAPS attendus : {n_corrects}/7 (nouveau) vs 5/7 (ancien)
""")
