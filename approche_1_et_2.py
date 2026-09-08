# -*- coding: utf-8 -*-
"""
Approches 1 et 2 : calcul de la similarité sans seuil calibré
=========================================================================

Version nettoyée du notebook original (le fichier source contenait
~1300 lignes, dont la grande majorité générait des figures d'une
version antérieure de l'article, avec l'ancien seuil 0,55 — retirées
ici pour ne garder que le calcul lui-même).

Approche 1 : ESCOXLM-R post-DAPT (modèle M3 sans sa tête NER)
Approche 2 : Sentence-BERT (MiniLM), sans seuil de décision

Ces deux approches ont été calculées avec l'encodeur MiniLM, avant le
diagnostic qui a motivé le passage à mpnet pour la version finale
(cf. diagnostic_top1_pare_feu.py). Les écarts de discrimination
obtenus ici (Approche 1 : écart=0,02 ; Approche 2 : écart=0,26) sont
cités dans l'article comme mesures préliminaires, et leur stabilité
est confirmée par la remesure ultérieure de l'Approche 2 avec mpnet
(écart=0,24, cf. recalibration_mpnet.py et Section 4.6 de l'article).
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================================
# DONNÉES COMMUNES AUX DEUX APPROCHES
# =========================================================================

df_entites = pd.read_csv("/content/drive/MyDrive/alignement/entites_test_gaps.csv")
competences = df_entites['texte_entite'].unique().tolist()

df_pe = pd.read_csv("/content/drive/MyDrive/alignement/PE_SN_segments_harmonized.csv")
segments_pe = df_pe['texte_source'].tolist()
modules_pe = df_pe['module_id'].tolist()

print(f"Compétences à aligner : {len(competences)}")
print(f"Segments PE : {len(segments_pe)}")


def aligner_par_similarite(competences, segments_pe, modules_pe, fn_embedding):
    """Calcule, pour chaque compétence, le meilleur segment PE (top-1)
    selon une fonction d'embedding donnée."""
    resultats = []
    for i, comp in enumerate(competences):
        if (i + 1) % 10 == 0:
            print(f"   Progression : {i+1}/{len(competences)}")
        emb_comp = fn_embedding(comp)
        best_sim, best_seg, best_mod = 0, "", ""
        for seg, mod in zip(segments_pe, modules_pe):
            emb_seg = fn_embedding(seg)
            sim = cosine_similarity(emb_comp, emb_seg)[0][0]
            if sim > best_sim:
                best_sim, best_seg, best_mod = sim, seg, mod
        resultats.append({
            'competence': comp,
            'similarite_max': round(best_sim, 4),
            'meilleur_match': best_seg,
            'module': best_mod
        })
    return pd.DataFrame(resultats)


# =========================================================================
# APPROCHE 1 : ESCOXLM-R POST-DAPT + COSINUS SIMPLE
# =========================================================================

print("=" * 70)
print("APPROCHE 1 : ESCOXLM-R POST-DAPT + COSINUS SIMPLE")
print("=" * 70)

from transformers import AutoTokenizer, AutoModel
import torch

model_path = "/content/drive/MyDrive/corpusdz/ESCOXLM-R_NER_M3"
tokenizer_escoxlm = AutoTokenizer.from_pretrained(model_path)
model_escoxlm = AutoModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_escoxlm.to(device)
model_escoxlm.eval()
print("ESCOXLM-R post-DAPT chargé")


def get_embedding_escoxlm(texte):
    """Embedding ESCOXLM-R par mean pooling (sans tête de classification NER)."""
    inputs = tokenizer_escoxlm(texte, return_tensors="pt",
                                truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model_escoxlm(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()


df_ap1 = aligner_par_similarite(competences, segments_pe, modules_pe, get_embedding_escoxlm)

print(f"\nRÉSULTATS APPROCHE 1 (ESCOXLM-R post-DAPT)")
print(f"   Similarité moyenne : {df_ap1['similarite_max'].mean():.4f}")
print(f"   Écart-type         : {df_ap1['similarite_max'].std():.4f}")

df_ap1.to_csv("/content/drive/MyDrive/alignement/resultats_approche1.csv", index=False)
print(f"Sauvegardé : resultats_approche1.csv")


# =========================================================================
# APPROCHE 2 : SENTENCE-BERT (MiniLM) + COSINUS SIMPLE, SANS SEUIL
# =========================================================================

print("\n" + "=" * 70)
print("APPROCHE 2 : Sentence-BERT (MiniLM) + COSINUS SIMPLE (SANS SEUIL)")
print("=" * 70)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Sentence-BERT (MiniLM) chargé")


def get_embedding_sbert(texte):
    return model.encode([texte])


df_ap2 = aligner_par_similarite(competences, segments_pe, modules_pe, get_embedding_sbert)

print(f"\nRÉSULTATS APPROCHE 2 (Sentence-BERT MiniLM, sans seuil)")
print(f"   Similarité moyenne : {df_ap2['similarite_max'].mean():.4f}")

# Sans seuil calibré, on ne peut objectivement pas décider — illustration
# de ce que donnerait un seuil arbitraire non validé :
seuil_arbitraire = 0.5
nb_alignes = len(df_ap2[df_ap2['similarite_max'] >= seuil_arbitraire])
print(f"\n   Avec un seuil ARBITRAIRE de {seuil_arbitraire} (non validé) :")
print(f"   → {nb_alignes}/{len(df_ap2)} seraient ALIGNÉS, sans justification scientifique")

df_ap2.to_csv("/content/drive/MyDrive/alignement/resultats_approche2.csv", index=False)
print(f"Sauvegardé : resultats_approche2.csv")


# =========================================================================
# ÉCART DE DISCRIMINATION GAPS / NON-GAPS (chiffre cité dans l'article)
# =========================================================================

gaps_attendus = ['docker', 'python', 'zabbix', 'cloud', 'nessus', 'siem',
                  'stormshield']


def calculer_ecart(df, col_sim='similarite_max'):
    """Écart entre la similarité moyenne des Non-GAPS et des GAPS —
    mesure la capacité du modèle à discriminer les deux catégories."""
    est_gap = df['competence'].str.lower().apply(
        lambda c: any(g in c for g in gaps_attendus)
    )
    moy_gaps = df[est_gap][col_sim].mean()
    moy_non_gaps = df[~est_gap][col_sim].mean()
    return moy_non_gaps - moy_gaps, moy_gaps, moy_non_gaps


ecart_ap1, gaps_ap1, nongaps_ap1 = calculer_ecart(df_ap1)
ecart_ap2, gaps_ap2, nongaps_ap2 = calculer_ecart(df_ap2)

print("\n" + "=" * 70)
print("ÉCART DE DISCRIMINATION GAPS/NON-GAPS")
print("=" * 70)
print(f"   Approche 1 (ESCOXLM-R+DAPT) : écart = {ecart_ap1:.4f} "
      f"(GAPS={gaps_ap1:.4f}, Non-GAPS={nongaps_ap1:.4f})")
print(f"   Approche 2 (Sentence-BERT MiniLM) : écart = {ecart_ap2:.4f} "
      f"(GAPS={gaps_ap2:.4f}, Non-GAPS={nongaps_ap2:.4f})")
print(f"\n   → Cité dans l'article : écart Approche 1 = 0.02, "
      f"écart Approche 2 (préliminaire, MiniLM) = 0.26")
