"""
Diagnostic systématique : recalcul complet du Groupe 1
=====================================================================

Recalcule le top-1 pour LES 106 COMPÉTENCES (pas un seul cas isolé),
avec un calcul par lots (batch) plus rapide et plus fiable que la
boucle imbriquée d'origine, puis compare au résultat originalement
enregistré. Ça révèle l'ampleur réelle du problème : est-ce que
"pare-feu" est un cas isolé, ou un symptôme d'un souci plus large ?
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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


model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✅ Modèle chargé")

# --- Référentiel PE ---
df_pe = pd.read_csv("/content/drive/MyDrive/alignement/PE_SN_segments_harmonized.csv")
segments_pe = df_pe['texte_source'].tolist()
modules_pe = df_pe['module_id'].tolist()

# --- Original alignment results (initial threshold 0.55) ---
df_original = lire_csv_robuste(
    "/content/drive/MyDrive/alignement/resultats_alignement_corrige.csv"
)

# =========================================================================
# RECALCUL PAR LOTS (batch) — beaucoup plus rapide ET plus fiable que
# la boucle imbriquée d'origine (qui ré-encodait chaque segment PE à
# chaque itération, une source classique d'erreurs/incohérences)
# =========================================================================

competences = df_original['competence_marche'].tolist()

print(f"Encodage de {len(competences)} compétences et {len(segments_pe)} segments PE...")
emb_competences = model.encode(competences, show_progress_bar=True)
emb_segments = model.encode(segments_pe, show_progress_bar=True)

# Matrice complète (106 x 312) en une seule opération
matrice_sim = cosine_similarity(emb_competences, emb_segments)

top1_idx = np.argmax(matrice_sim, axis=1)
top1_score = np.max(matrice_sim, axis=1)
top1_segment = [segments_pe[i] for i in top1_idx]
top1_module = [modules_pe[i] for i in top1_idx]

df_recalcule = pd.DataFrame({
    'competence_marche': competences,
    'similarite_recalculee': np.round(top1_score, 4),
    'segment_recalcule': top1_segment,
    'module_recalcule': top1_module,
})

# =========================================================================
# COMPARAISON : original vs recalculé
# =========================================================================

df_comp = df_original[['competence_marche', 'similarite_max', 'meilleur_match_pe']].merge(
    df_recalcule, on='competence_marche', how='inner'
)

df_comp['ecart_score'] = (df_comp['similarite_recalculee'] - df_comp['similarite_max']).round(4)
df_comp['segment_different'] = df_comp['meilleur_match_pe'] != df_comp['segment_recalcule']

print("\n" + "=" * 70)
print("RÉSUMÉ DE LA COMPARAISON")
print("=" * 70)
print(f"Total compétences comparées : {len(df_comp)}")
print(f"Segments DIFFÉRENTS trouvés (original vs recalcul) : {df_comp['segment_different'].sum()}")
print(f"Écart moyen de score : {df_comp['ecart_score'].mean():.4f}")
print(f"Écart max de score   : {df_comp['ecart_score'].abs().max():.4f}")

# Cas où le recalcul trouve un score NETTEMENT meilleur (>0.05 d'écart)
# avec un segment différent = signal fort d'un problème dans le calcul original
suspects = df_comp[(df_comp['segment_different']) & (df_comp['ecart_score'] > 0.05)]
suspects = suspects.sort_values('ecart_score', ascending=False)

print(f"\n⚠️ Cas SUSPECTS (segment différent ET score recalculé nettement plus haut, >0.05) : {len(suspects)}")
print("=" * 70)
for _, row in suspects.iterrows():
    print(f"\n  \"{row['competence_marche']}\"")
    print(f"    Original  : \"{row['meilleur_match_pe']}\" (sim={row['similarite_max']:.4f})")
    print(f"    Recalculé : \"{row['segment_recalcule']}\" (sim={row['similarite_recalculee']:.4f})")
    print(f"    Écart : +{row['ecart_score']:.4f}")

df_comp.to_csv(
    "/content/drive/MyDrive/alignement/diagnostic_comparaison_groupe1.csv",
    index=False, encoding='utf-8'
)

print(f"\n💾 Comparaison complète sauvegardée : diagnostic_comparaison_groupe1.csv")

print("\n" + "=" * 70)
print("INTERPRÉTATION")
print("=" * 70)
if len(suspects) > 5:
    print(f"""
{len(suspects)} cas suspects trouvés — ce n'est PAS un problème isolé
au pare-feu. Il y a probablement eu un bug systématique dans le calcul
d'alignement original (à investiguer : cache d'embeddings, exécution
partielle, ordre des segments...). Recommandation : ré-exécuter
l'alignement Groupe 1 ET Groupe 2 depuis zéro avec ce recalcul par
lots (plus fiable), puis reclasser avec le nouveau seuil 0.8257
sur ces scores fraîchement recalculés.
""")
else:
    print(f"""
Seulement {len(suspects)} cas suspects — le pare-feu semble être une
exception plutôt que la règle. Les résultats du seuil 0.8257 sur les
scores originaux restent globalement fiables, avec quelques cas
isolés à corriger manuellement.
""")
