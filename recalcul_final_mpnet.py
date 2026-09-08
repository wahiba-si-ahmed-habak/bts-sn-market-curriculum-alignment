"""
Recalcul FINAL Groupe 1 et Groupe 2 — mpnet + seuil 0.7858
=====================================================================

Version définitive : recalcule le top-1 pour toutes les compétences
des deux groupes avec mpnet (le modèle qui a résolu le cas du
pare-feu), puis classe ALIGNE/GAP avec le seuil calibré sur ce même
modèle (0.7858, exactitude 86%, F1=0.863 sur les 50 paires
consensuelles).
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


SEUIL_FINAL = 0.7858

print("Chargement de paraphrase-multilingual-mpnet-base-v2...")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
print("✅ Modèle chargé")

df_pe = pd.read_csv("/content/drive/MyDrive/alignement/PE_SN_segments_harmonized.csv")
segments_pe = df_pe['texte_source'].tolist()
modules_pe = df_pe['module_id'].tolist()
print(f"✅ {len(segments_pe)} segments PE chargés")

emb_segments_pe = model.encode(segments_pe, show_progress_bar=True)


def aligner_groupe(competences, seuil, nom_groupe):
    """Calcule le top-1 (mpnet) pour chaque compétence et classe ALIGNE/GAP."""
    print(f"\nEncodage de {len(competences)} compétences ({nom_groupe})...")
    emb_competences = model.encode(competences, show_progress_bar=True)

    matrice_sim = cosine_similarity(emb_competences, emb_segments_pe)
    top1_idx = np.argmax(matrice_sim, axis=1)
    top1_score = np.max(matrice_sim, axis=1)

    df_res = pd.DataFrame({
        'competence': competences,
        'similarite': np.round(top1_score, 4),
        'meilleur_match_pe': [segments_pe[i] for i in top1_idx],
        'module_pe': [modules_pe[i] for i in top1_idx],
    })
    df_res['statut'] = df_res['similarite'].apply(lambda s: 'ALIGNE' if s >= seuil else 'GAP')
    return df_res


# =========================================================================
# GROUPE 1 (8 offres, mêmes 106 compétences filtrées qu'avant)
# =========================================================================

print("=" * 70)
print("GROUPE 1")
print("=" * 70)

df_g1_original = lire_csv_robuste(
    "/content/drive/MyDrive/alignement/resultats_alignement_corrige.csv"
)
competences_g1 = df_g1_original['competence_marche'].tolist()

df_g1_final = aligner_groupe(competences_g1, SEUIL_FINAL, "Groupe 1")

n_alignes_g1 = (df_g1_final['statut'] == 'ALIGNE').sum()
n_total_g1 = len(df_g1_final)
print(f"\n📊 GROUPE 1 — RÉSULTAT FINAL (mpnet, seuil {SEUIL_FINAL})")
print(f"   ALIGNÉS : {n_alignes_g1}/{n_total_g1} ({n_alignes_g1/n_total_g1*100:.1f}%)")
print(f"   GAPS    : {n_total_g1-n_alignes_g1}/{n_total_g1} ({(n_total_g1-n_alignes_g1)/n_total_g1*100:.1f}%)")

print(f"\n❌ GAPS détectés :")
for _, row in df_g1_final[df_g1_final['statut']=='GAP'].sort_values('similarite', ascending=False).iterrows():
    print(f"   • \"{row['competence']}\" (sim: {row['similarite']:.3f}) → {row['meilleur_match_pe']} [{row['module_pe']}]")

df_g1_final.to_csv(
    "/content/drive/MyDrive/alignement/resultats_groupe1_MPNET_FINAL.csv",
    index=False, encoding='utf-8'
)

# =========================================================================
# GROUPE 2 (2 offres test, 51 compétences, 7 GAPS volontaires attendus)
# =========================================================================

print("\n" + "=" * 70)
print("GROUPE 2")
print("=" * 70)

df_g2_original = lire_csv_robuste(
    "/content/drive/MyDrive/alignement/resultats_test_gaps.csv"
)
competences_g2 = df_g2_original['competence'].tolist()

df_g2_final = aligner_groupe(competences_g2, SEUIL_FINAL, "Groupe 2")

n_alignes_g2 = (df_g2_final['statut'] == 'ALIGNE').sum()
n_total_g2 = len(df_g2_final)
print(f"\n📊 GROUPE 2 — RÉSULTAT FINAL (mpnet, seuil {SEUIL_FINAL})")
print(f"   ALIGNÉS : {n_alignes_g2}/{n_total_g2} ({n_alignes_g2/n_total_g2*100:.1f}%)")
print(f"   GAPS    : {n_total_g2-n_alignes_g2}/{n_total_g2} ({(n_total_g2-n_alignes_g2)/n_total_g2*100:.1f}%)")

# Vérification sur les 7 GAPS volontaires
print(f"\n🔍 VÉRIFICATION SUR LES 7 GAPS ATTENDUS :")
attendus_gap = ['docker', 'python', 'zabbix', 'cloud', 'nessus', 'siem', 'stormshield']
df_ok = df_g2_final[df_g2_final['statut'] == 'ALIGNE']
df_gap = df_g2_final[df_g2_final['statut'] == 'GAP']
n_corrects = 0
for terme in attendus_gap:
    dans_gap = any(terme in c.lower() for c in df_gap['competence'])
    dans_ok = any(terme in c.lower() for c in df_ok['competence'])
    if dans_gap:
        print(f"   ✅ \"{terme}\" → GAP (correct)")
        n_corrects += 1
    elif dans_ok:
        print(f"   ⚠️ \"{terme}\" → ALIGNÉ (problème)")
    else:
        print(f"   ❓ \"{terme}\" → pas extrait")
print(f"\n   Score de détection : {n_corrects}/7 ({n_corrects/7*100:.1f}%)")

df_g2_final.to_csv(
    "/content/drive/MyDrive/alignement/resultats_groupe2_MPNET_FINAL.csv",
    index=False, encoding='utf-8'
)

# =========================================================================
# RÉCAPITULATIF COMPLET POUR L'ARTICLE
# =========================================================================

print("\n" + "=" * 70)
print("📋 RÉCAPITULATIF COMPLET (à utiliser dans l'article)")
print("=" * 70)
print(f"""
Modèle          : paraphrase-multilingual-mpnet-base-v2
Seuil calibré   : {SEUIL_FINAL} (κ inter-annotateurs = 0.717 ; exactitude
                  calibration = 86.0% ; F1 = 0.863 ; Mann-Whitney p<0.000001)

Groupe 1 (8 offres, {n_total_g1} compétences)  : {n_alignes_g1}/{n_total_g1} alignées ({n_alignes_g1/n_total_g1*100:.1f}%)
Groupe 2 (2 offres test, {n_total_g2} compétences) : {n_alignes_g2}/{n_total_g2} alignées ({n_alignes_g2/n_total_g2*100:.1f}%)
Détection des 7 GAPS volontaires             : {n_corrects}/7 ({n_corrects/7*100:.1f}%)
""")
