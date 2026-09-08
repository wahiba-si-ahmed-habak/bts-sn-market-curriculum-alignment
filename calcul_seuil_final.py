"""
Calcul du seuil optimal - Version finale (échantillon consensuel)
=====================================================================

Ce script :
  1. Résout les désaccords entre les 2 annotateurs (règle : NON_MATCH
     par défaut en cas de désaccord)
  2. Fusionne avec les scores de similarité (fichier _reference)
  3. Calcule le nouveau seuil optimal (midpoint des moyennes)
  4. Teste la significativité statistique (Mann-Whitney U)
  5. Évalue la performance (rappel, spécificité, exactitude, F1)

Fichiers nécessaires (générés par template_annotation_v2_paire_fixe.py) :
  - template_annotation_v2_aveugle.csv   (rempli par les 2 annotateurs)
  - template_annotation_v2_reference.csv (avec les scores de similarité)
"""

import pandas as pd
import numpy as np
from scipy import stats


# =========================================================================
# 1. RÉSOLUTION DES DÉSACCORDS (NON_MATCH par défaut)
# =========================================================================

def resoudre_consensus(df, regle="non_match_par_defaut"):
    """
    Construit le label final à partir des 2 annotateurs.

    Paramètres
    ----------
    df : DataFrame avec colonnes type_pair_annotateur1 et type_pair_annotateur2
    regle : str
        "non_match_par_defaut" : en cas de désaccord, NON_MATCH l'emporte
        (règle conservatrice : on préfère rater un MATCH plutôt que
        valider un mauvais alignement)
    """
    def _consensus(row):
        if row['type_pair_annotateur1'] == row['type_pair_annotateur2']:
            return row['type_pair_annotateur1']
        if regle == "non_match_par_defaut":
            return 'NON_MATCH'
        raise ValueError(f"Règle inconnue : {regle}")

    df = df.copy()
    df['type_pair_final'] = df.apply(_consensus, axis=1)
    df['etait_desaccord'] = df['type_pair_annotateur1'] != df['type_pair_annotateur2']
    return df


# =========================================================================
# 2. CALCUL DU SEUIL OPTIMAL
# =========================================================================

def calculer_seuil_optimal(scores_match, scores_non_match):
    """Seuil optimal = midpoint entre les deux moyennes."""
    scores_match = np.array(scores_match)
    scores_non_match = np.array(scores_non_match)

    stats_desc = {
        "n_match": len(scores_match),
        "n_non_match": len(scores_non_match),
        "mean_match": round(float(scores_match.mean()), 4),
        "median_match": round(float(np.median(scores_match)), 4),
        "min_match": round(float(scores_match.min()), 4),
        "max_match": round(float(scores_match.max()), 4),
        "mean_non_match": round(float(scores_non_match.mean()), 4),
        "median_non_match": round(float(np.median(scores_non_match)), 4),
        "min_non_match": round(float(scores_non_match.min()), 4),
        "max_non_match": round(float(scores_non_match.max()), 4),
    }
    stats_desc["ecart_moyennes"] = round(
        stats_desc["mean_match"] - stats_desc["mean_non_match"], 4
    )
    stats_desc["chevauchement"] = round(
        stats_desc["max_non_match"] - stats_desc["min_match"], 4
    )
    seuil_optimal = (scores_match.mean() + scores_non_match.mean()) / 2
    stats_desc["seuil_optimal"] = round(float(seuil_optimal), 4)

    return stats_desc


def test_significativite(scores_match, scores_non_match):
    """Mann-Whitney U entre les deux groupes."""
    u_stat, p_value = stats.mannwhitneyu(
        scores_match, scores_non_match, alternative="two-sided"
    )
    return {
        "U_statistic": round(float(u_stat), 2),
        "p_value": round(float(p_value), 6),
        "significatif_0.05": p_value < 0.05,
    }


def evaluer_performance(seuil, scores_match, scores_non_match):
    """Rappel, spécificité, exactitude, précision, F1 à un seuil donné."""
    scores_match = np.array(scores_match)
    scores_non_match = np.array(scores_non_match)

    vp = int((scores_match >= seuil).sum())
    fn = int((scores_match < seuil).sum())
    vn = int((scores_non_match < seuil).sum())
    fp = int((scores_non_match >= seuil).sum())

    rappel = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    specificite = vn / (vn + fp) if (vn + fp) > 0 else 0.0
    exactitude = (vp + vn) / (vp + fn + vn + fp)
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    f1 = (2 * precision * rappel / (precision + rappel)
          if (precision + rappel) > 0 else 0.0)

    return {
        "seuil": seuil, "VP": vp, "FN": fn, "VN": vn, "FP": fp,
        "rappel": round(rappel, 4), "specificite": round(specificite, 4),
        "exactitude": round(exactitude, 4), "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


def lire_csv_robuste(chemin):
    """
    Lit un CSV en détectant automatiquement l'encodage (UTF-8 ou cp1252)
    et le séparateur (virgule ou point-virgule) — évite les erreurs
    UnicodeDecodeError fréquentes avec des fichiers exportés depuis Excel.
    """
    for encoding in ['utf-8', 'cp1252', 'latin-1']:
        for sep in [',', ';']:
            try:
                df = pd.read_csv(chemin, encoding=encoding, sep=sep)
                if len(df.columns) > 1:  # bon séparateur si plusieurs colonnes détectées
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
    raise ValueError(f"Impossible de lire {chemin} avec les encodages/séparateurs testés")


# =========================================================================
# EXÉCUTION
# =========================================================================

if __name__ == "__main__":

    # --- Chargement depuis Google Drive ---
    df_aveugle = lire_csv_robuste("/content/drive/MyDrive/alignement/template_annotation_v2_aveugle.csv")
    df_reference = lire_csv_robuste("/content/drive/MyDrive/alignement/template_annotation_v2_reference.csv")

    # --- Consensus ---
    df_aveugle = resoudre_consensus(df_aveugle, regle="non_match_par_defaut")
    print(f"Répartition finale : {df_aveugle['type_pair_final'].value_counts().to_dict()}")
    print(f"Désaccords résolus : {df_aveugle['etait_desaccord'].sum()} / {len(df_aveugle)}")

    # --- Fusion avec les scores ---
    df = df_aveugle[['id_paire', 'type_pair_final']].merge(
        df_reference[['id_paire', 'similarite_cosinus']], on='id_paire', how='left'
    )
    assert df['similarite_cosinus'].isnull().sum() == 0, "Scores manquants après fusion !"

    scores_match = df[df['type_pair_final'] == 'MATCH']['similarite_cosinus'].values
    scores_non_match = df[df['type_pair_final'] == 'NON_MATCH']['similarite_cosinus'].values

    # --- Calibration ---
    print("\n" + "=" * 70)
    print("STATISTIQUES DE CALIBRATION")
    print("=" * 70)
    desc = calculer_seuil_optimal(scores_match, scores_non_match)
    for k, v in desc.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("TEST DE SIGNIFICATIVITÉ (Mann-Whitney U)")
    print("=" * 70)
    sig = test_significativite(scores_match, scores_non_match)
    for k, v in sig.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print(f"PERFORMANCE AU NOUVEAU SEUIL OPTIMAL ({desc['seuil_optimal']})")
    print("=" * 70)
    perf = evaluer_performance(desc['seuil_optimal'], scores_match, scores_non_match)
    for k, v in perf.items():
        print(f"  {k}: {v}")

    print("\n>>> Phrase type pour l'article :")
    print(
        f'"Le nouveau seuil, calibré sur {desc["n_match"]+desc["n_non_match"]} paires '
        f'consensuelles (accord inter-annotateurs κ=0.717), est fixé à '
        f'{desc["seuil_optimal"]}, atteignant une exactitude de '
        f'{perf["exactitude"]*100:.1f}%, une spécificité de '
        f'{perf["specificite"]*100:.1f}% et un rappel de {perf["rappel"]*100:.1f}% '
        f'(Mann-Whitney U, p={sig["p_value"]})."'
    )
