"""
Recalibration avec mpnet — réutilise les labels déjà annotés
=====================================================================

Les 50 paires ont déjà été jugées MATCH/NON_MATCH par 2 humains
(kappa=0.717). Ce jugement porte sur le CONTENU des paires, pas sur
le modèle qui les a suggérées — donc on peut réutiliser ces mêmes
labels et juste recalculer leur score de similarité avec mpnet.

Étapes :
  1. Recharge le template annoté (aveugle) + résout les désaccords
     (NON_MATCH par défaut, comme convenu)
  2. Recalcule la similarité mpnet pour ces 50 MÊMES paires
     (même compétence, même segment_pe_candidat)
  3. Recalibre le seuil optimal sur ces nouveaux scores
  4. Évalue la performance
"""

import pandas as pd
import numpy as np
from scipy import stats
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


def resoudre_consensus(df, regle="non_match_par_defaut"):
    def _consensus(row):
        if row['type_pair_annotateur1'] == row['type_pair_annotateur2']:
            return row['type_pair_annotateur1']
        return 'NON_MATCH'
    df = df.copy()
    df['type_pair_final'] = df.apply(_consensus, axis=1)
    return df


# =========================================================================
# 1. CONSENSUS (identique à avant)
# =========================================================================

df_aveugle = lire_csv_robuste(
    "/content/drive/MyDrive/alignement/template_annotation_v2_aveugle.csv"
)
df_aveugle = resoudre_consensus(df_aveugle)
print(f"Consensus : {df_aveugle['type_pair_final'].value_counts().to_dict()}")

# =========================================================================
# 2. RECALCUL DES SCORES AVEC MPNET (mêmes paires, nouveau modèle)
# =========================================================================

print("\nChargement de paraphrase-multilingual-mpnet-base-v2...")
model_fort = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
print("✅ Modèle chargé")

competences = df_aveugle['competence_marche'].tolist()
segments = df_aveugle['segment_pe_candidat'].tolist()

emb_comp = model_fort.encode(competences, show_progress_bar=True)
emb_seg = model_fort.encode(segments, show_progress_bar=True)

# Similarité PAIRE PAR PAIRE (pas une matrice complète — on compare
# chaque compétence à SON segment déjà fixé, pas à tous les segments)
scores_mpnet = [
    cosine_similarity([emb_comp[i]], [emb_seg[i]])[0][0]
    for i in range(len(competences))
]
df_aveugle['similarite_mpnet'] = np.round(scores_mpnet, 4)

# =========================================================================
# 3. RECALIBRATION DU SEUIL
# =========================================================================

scores_match = df_aveugle[df_aveugle['type_pair_final'] == 'MATCH']['similarite_mpnet'].values
scores_non_match = df_aveugle[df_aveugle['type_pair_final'] == 'NON_MATCH']['similarite_mpnet'].values

print("\n" + "=" * 70)
print("STATISTIQUES DE CALIBRATION (mpnet)")
print("=" * 70)
print(f"N MATCH: {len(scores_match)}, N NON_MATCH: {len(scores_non_match)}")
print(f"MATCH     - moyenne: {scores_match.mean():.4f}, médiane: {np.median(scores_match):.4f}")
print(f"NON_MATCH - moyenne: {scores_non_match.mean():.4f}, médiane: {np.median(scores_non_match):.4f}")

seuil_optimal = (scores_match.mean() + scores_non_match.mean()) / 2
print(f"\nNouveau seuil optimal (mpnet) : {seuil_optimal:.4f}")

u_stat, p_value = stats.mannwhitneyu(scores_match, scores_non_match, alternative='two-sided')
print(f"Mann-Whitney U : U={u_stat:.2f}, p={p_value:.6f} (significatif: {p_value < 0.05})")

# =========================================================================
# 4. PERFORMANCE
# =========================================================================

def evaluer(seuil, scores_match, scores_non_match):
    vp = int((scores_match >= seuil).sum())
    fn = int((scores_match < seuil).sum())
    vn = int((scores_non_match < seuil).sum())
    fp = int((scores_non_match >= seuil).sum())
    rappel = vp / (vp + fn) if (vp + fn) > 0 else 0
    specificite = vn / (vn + fp) if (vn + fp) > 0 else 0
    exactitude = (vp + vn) / (vp + fn + vn + fp)
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0
    f1 = 2 * precision * rappel / (precision + rappel) if (precision + rappel) > 0 else 0
    return dict(rappel=rappel, specificite=specificite, exactitude=exactitude, precision=precision, f1=f1)

perf = evaluer(seuil_optimal, scores_match, scores_non_match)
print(f"\nPerformance au seuil {seuil_optimal:.4f} :")
for k, v in perf.items():
    print(f"  {k}: {v:.4f}")

df_aveugle.to_csv(
    "/content/drive/MyDrive/alignement/calibration_mpnet_finale.csv",
    index=False, encoding='utf-8'
)
print(f"\n💾 Sauvegardé : calibration_mpnet_finale.csv")
print(f"\n👉 SEUIL À UTILISER POUR LE RECALCUL GROUPE 1/2 : {seuil_optimal:.4f}")
