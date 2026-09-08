"""
Template de double annotation CORRIGÉ — paire fixe pour un kappa valide
==========================================================================

Différence avec la version précédente : au lieu de laisser chaque
annotateur chercher librement son propre segment_pe, ce script calcule
UN SEUL candidat (le plus proche par similarité cosinus, comme le fait
le modèle final) et demande aux DEUX annotateurs de juger CETTE MÊME
paire. C'est la seule façon d'obtenir un Cohen's Kappa interprétable.

À utiliser dans Colab, avec le même modèle et les mêmes fichiers que
votre pipeline existant.
"""

import pandas as pd
import numpy as np
import re
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================================
# 1. CHARGEMENT (mêmes fichiers que votre pipeline)
# =========================================================================

df_pe = pd.read_csv("/content/drive/MyDrive/alignement/PE_SN_segments_harmonized.csv")
segments_pe = df_pe['texte_source'].tolist()
modules_pe = df_pe['module_id'].tolist()

df_entites = pd.read_csv("/content/drive/MyDrive/alignement/entites_extraites_m3.csv")

print(f"✅ {len(segments_pe)} segments PE chargés")

# =========================================================================
# 2. NETTOYAGE + FILTRAGE RENFORCÉ (identique au script précédent)
# =========================================================================

def nettoyer_entite(texte):
    texte = str(texte).strip()
    texte = re.sub(r'\s+', ' ', texte)
    texte = texte.strip('.,;:!?()[]{}«»""\'- ')
    return texte

df_entites['texte_entite_clean'] = df_entites['texte_entite'].apply(nettoyer_entite)
df_comp = df_entites[df_entites['type_entite'].isin(['SKILL', 'KNOW'])].copy()

DIPLOMES = {'bts', 'licence', 'licenc', 'master', 'certification', 'certif',
            'bac', 'baccalauréat', 'ccna', 'ccnp'}
MOTS_FIN_INVALIDE = {'les', 'des', 'de', 'du', 'la', 'le', 'un', 'une',
                      'et', 'ou', 'sur', 'sous', 'avec', 'pour', 'dans'}
ACRONYMES_VALIDES = {'dns', 'dhcp', 'vpn', 'acl', 'lan', 'wan', 'ids',
                      'ips', 'aaa', 'ssh', 'ios', 'sid', 'nas', 'gpo',
                      'pki', 'pap', 'chap', 'lcp', 'ncp', 'ppp', 'dsl'}

def est_valide_v2(texte, type_entite):
    texte_lower = texte.lower().strip()
    mots = texte.split()
    if len(texte) < 3:
        return False
    if texte_lower in DIPLOMES:
        return False
    dernier_mot = mots[-1].lower() if mots else ""
    if dernier_mot in MOTS_FIN_INVALIDE:
        return False
    if len(texte) <= 4 and texte_lower not in ACRONYMES_VALIDES and texte_lower.isalpha():
        return False
    if type_entite == 'SKILL' and len(mots) < 2:
        return False
    return True

df_comp['valide'] = df_comp.apply(
    lambda row: est_valide_v2(row['texte_entite_clean'], row['type_entite']), axis=1
)
df_valide_unique = df_comp[df_comp['valide']].drop_duplicates(subset=['texte_entite_clean'])

# =========================================================================
# 3. TOUTES LES COMPÉTENCES VALIDES (on calcule le top-1 sur l'ensemble
#    avant de sélectionner, pour pouvoir stratifier par score ensuite)
# =========================================================================

SEED = 2026
random.seed(SEED)
competences_pool = df_valide_unique['texte_entite_clean'].tolist()

print(f"✅ {len(competences_pool)} compétences valides disponibles (pool complet)")

# =========================================================================
# 4. CALCUL DU TOP-1 SEGMENT PE PAR SIMILARITÉ COSINUS, SUR TOUT LE POOL
#    (même modèle que votre pipeline : paraphrase-multilingual-MiniLM-L12-v2)
# =========================================================================

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✅ Modèle Sentence-BERT chargé")

emb_competences_pool = model.encode(competences_pool, show_progress_bar=True)
emb_segments_pe = model.encode(segments_pe, show_progress_bar=True)

sim_matrix_pool = cosine_similarity(emb_competences_pool, emb_segments_pe)

top1_idx_pool = np.argmax(sim_matrix_pool, axis=1)
top1_score_pool = np.max(sim_matrix_pool, axis=1)

df_pool = pd.DataFrame({
    'competence': competences_pool,
    'segment_pe': [segments_pe[i] for i in top1_idx_pool],
    'module_pe': [modules_pe[i] for i in top1_idx_pool],
    'score': top1_score_pool,
}).sort_values('score', ascending=False).reset_index(drop=True)

# =========================================================================
# 5. SÉLECTION STRATIFIÉE SUR TOUTE LA GAMME DES SCORES (déciles)
#    Plutôt que de prendre seulement les extrêmes (trop faciles), on
#    répartit l'échantillon sur toute la distribution des scores, avec
#    une bonne couverture de la zone médiane (proche de la frontière de
#    décision) où se joue vraiment la calibration du seuil.
#    ⚠️ À décrire dans l'article comme "échantillonnage stratifié par
#    décile de score de similarité", PAS comme un tirage aléatoire simple.
# =========================================================================

N_TOTAL = 50
N_DECILES = 10
N_PAR_DECILE = N_TOTAL // N_DECILES  # 5 par décile

df_pool['decile'] = pd.qcut(df_pool['score'], q=N_DECILES, labels=False, duplicates='drop')

# Boucle explicite + concat (plus robuste que groupby.apply selon la version pandas)
morceaux = []
for d in sorted(df_pool['decile'].unique()):
    sous_groupe = df_pool[df_pool['decile'] == d]
    n = min(N_PAR_DECILE, len(sous_groupe))
    morceaux.append(sous_groupe.sample(n=n, random_state=SEED))

df_selection = pd.concat(morceaux).sample(frac=1, random_state=SEED).reset_index(drop=True)

competences_annotation = df_selection['competence'].tolist()
top1_segments = df_selection['segment_pe'].tolist()
top1_modules = df_selection['module_pe'].tolist()
top1_scores = df_selection['score'].tolist()

print(f"✅ Sélection stratifiée sur {N_DECILES} déciles : {len(competences_annotation)} paires au total")
print(f"   Score min sélectionné : {min(top1_scores):.3f}")
print(f"   Score max sélectionné : {max(top1_scores):.3f}")
print(f"\n   Répartition par décile (0=scores les plus bas, 9=les plus hauts) :")
print(df_selection['decile'].value_counts().sort_index())

# =========================================================================
# 5. TEMPLATE POUR ANNOTATION AVEUGLE (sans le score, pour éviter le biais)
# =========================================================================

template_aveugle = pd.DataFrame({
    'id_paire': [f'PAIRE_{i+1:03d}' for i in range(len(competences_annotation))],
    'competence_marche': competences_annotation,
    'segment_pe_candidat': top1_segments,       # MÊME candidat pour les 2 annotateurs
    'module_pe_candidat': top1_modules,
    'type_pair_annotateur1': '',                # à remplir : MATCH ou NON_MATCH
    'type_pair_annotateur2': '',                # à remplir : MATCH ou NON_MATCH
})

path_aveugle = "/content/drive/MyDrive/alignement/template_annotation_v2_aveugle.csv"
template_aveugle.to_csv(path_aveugle, index=False, encoding='utf-8')

# Version avec le score, gardée à part pour VOTRE usage (analyse a posteriori),
# à ne PAS montrer aux annotateurs pendant qu'ils jugent (évite l'ancrage)
template_reference = template_aveugle.copy()
template_reference['similarite_cosinus'] = np.round(top1_scores, 4)
path_reference = "/content/drive/MyDrive/alignement/template_annotation_v2_reference.csv"
template_reference.to_csv(path_reference, index=False, encoding='utf-8')

print(f"\n💾 Template annotateurs (sans score) : {path_aveugle}")
print(f"💾 Référence avec scores (pour vous)  : {path_reference}")
print(f"\n👉 Envoyez le fichier '_aveugle' aux deux annotateurs.")
print(f"   Chacun remplit UNIQUEMENT sa colonne type_pair_annotateurX,")
print(f"   sur la MÊME paire compétence/segment déjà fixée.")

# Aperçu
print(f"\n📋 APERÇU DES 10 PREMIÈRES PAIRES")
print("=" * 70)
for i in range(min(10, len(competences_annotation))):
    print(f"{i+1}. \"{competences_annotation[i]}\" → \"{top1_segments[i]}\" "
          f"({top1_modules[i]}, sim={top1_scores[i]:.3f})")
