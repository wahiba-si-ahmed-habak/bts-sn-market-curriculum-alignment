"""
Nouvel échantillon de calibration - Version propre avec double annotation
===========================================================================

Remplace le tirage aléatoire initial. Reprend les mêmes 151 entités déjà
extraites et nettoyées, mais avec :
  - Un seed documenté et assumé (2026, pas 42) pour marquer clairement
    qu'il s'agit d'un nouveau tirage, distinct de l'ancien
  - Un filtrage renforcé qui élimine les fragments d'extraction
    (diplômes isolés, phrases tronquées finissant par un article/préposition)
  - Une préparation prête pour DEUX annotateurs indépendants

À utiliser dans Colab, en remplacement du tirage initial.
"""

import pandas as pd
import numpy as np
import re
import random

# =========================================================================
# 1. CHARGEMENT (identique au tirage initial)
# =========================================================================

df_entites = pd.read_csv("/content/drive/MyDrive/alignement/entites_extraites_m3.csv")

def nettoyer_entite(texte):
    texte = str(texte).strip()
    texte = re.sub(r'\s+', ' ', texte)
    texte = texte.strip('.,;:!?()[]{}«»""\'- ')
    return texte

df_entites['texte_entite_clean'] = df_entites['texte_entite'].apply(nettoyer_entite)
df_comp = df_entites[df_entites['type_entite'].isin(['SKILL', 'KNOW'])].copy()

# =========================================================================
# 2. FILTRAGE RENFORCÉ (nouveau — corrige les fragments repérés)
# =========================================================================

# Termes de diplôme/qualification : ce sont des OCC, pas des compétences
DIPLOMES = {'bts', 'licence', 'licenc', 'master', 'certification', 'certif',
            'bac', 'baccalauréat', 'ccna', 'ccnp'}

# Mots qui, en fin d'entité, signalent une phrase coupée
# (ex: "Tester les", "Encadrer les" → coupée avant le complément)
MOTS_FIN_INVALIDE = {'les', 'des', 'de', 'du', 'la', 'le', 'un', 'une',
                      'et', 'ou', 'sur', 'sous', 'avec', 'pour', 'dans'}

def est_valide_v2(texte, type_entite):
    """Validation renforcée d'une entité candidate."""
    texte_lower = texte.lower().strip()
    mots = texte.split()

    if len(texte) < 3:
        return False, "trop court"

    # Rejet des diplômes/qualifications (ce sont des OCC, pas des SKILL/KNOW)
    if texte_lower in DIPLOMES:
        return False, "diplôme/qualification, pas une compétence"

    # Rejet des phrases coupées (dernier mot = article/préposition)
    dernier_mot = mots[-1].lower() if mots else ""
    if dernier_mot in MOTS_FIN_INVALIDE:
        return False, "phrase tronquée (coupée avant le complément)"

    # Rejet des mots seuls visiblement fragmentaires (< 4 caractères,
    # tout en minuscules, pas un acronyme connu)
    ACRONYMES_VALIDES = {'dns', 'dhcp', 'vpn', 'acl', 'lan', 'wan', 'ids',
                          'ips', 'aaa', 'ssh', 'ios', 'sid', 'nas', 'gpo',
                          'pki', 'pap', 'chap', 'lcp', 'ncp', 'ppp', 'dsl'}
    if len(texte) <= 4 and texte_lower not in ACRONYMES_VALIDES and texte_lower.isalpha():
        return False, "fragment possible (mot court non reconnu comme acronyme)"

    # Règle originale : SKILL doit avoir au moins 2 mots
    if type_entite == 'SKILL' and len(mots) < 2:
        return False, "SKILL trop court (1 mot)"

    return True, "valide"

# Appliquer le filtrage renforcé
resultats_validation = df_comp.apply(
    lambda row: est_valide_v2(row['texte_entite_clean'], row['type_entite']),
    axis=1
)
df_comp['valide'] = [r[0] for r in resultats_validation]
df_comp['raison_rejet'] = [r[1] for r in resultats_validation]

df_valide = df_comp[df_comp['valide']].copy()
df_valide_unique = df_valide.drop_duplicates(subset=['texte_entite_clean'])

df_rejetes = df_comp[~df_comp['valide']]

print("=" * 70)
print("📊 FILTRAGE RENFORCÉ")
print("=" * 70)
print(f"   Entités SKILL+KNOW brutes : {len(df_comp)}")
print(f"   Rejetées                  : {len(df_rejetes)}")
print(f"   Valides et uniques        : {len(df_valide_unique)}")

if len(df_rejetes) > 0:
    print(f"\n   Détail des rejets :")
    for raison in df_rejetes['raison_rejet'].unique():
        n = len(df_rejetes[df_rejetes['raison_rejet'] == raison])
        exemples = df_rejetes[df_rejetes['raison_rejet'] == raison]['texte_entite_clean'].head(3).tolist()
        print(f"     - {raison} ({n}) : {exemples}")

# =========================================================================
# 3. NOUVEAU TIRAGE ALÉATOIRE — SEED DOCUMENTÉ = 2026
# =========================================================================

SEED = 2026  # Documenté explicitement : nouveau tirage, distinct de l'ancien
random.seed(SEED)

competences_uniques = df_valide_unique['texte_entite_clean'].tolist()
random.shuffle(competences_uniques)

N_ANNOTATION = min(50, len(competences_uniques))
competences_annotation = competences_uniques[:N_ANNOTATION]

print(f"\n📊 NOUVEAU TIRAGE (seed={SEED})")
print(f"   Compétences disponibles après filtrage : {len(competences_uniques)}")
print(f"   Sélectionnées pour double annotation    : {len(competences_annotation)}")

# =========================================================================
# 4. TEMPLATE POUR DEUX ANNOTATEURS INDÉPENDANTS
# =========================================================================

template = pd.DataFrame({
    'id_paire': [f'PAIRE_{i+1:03d}' for i in range(len(competences_annotation))],
    'competence_marche': competences_annotation,
    'segment_pe_annotateur1': '',
    'module_pe_annotateur1': '',
    'type_pair_annotateur1': '',   # MATCH ou NON_MATCH
    'segment_pe_annotateur2': '',
    'module_pe_annotateur2': '',
    'type_pair_annotateur2': '',   # MATCH ou NON_MATCH
})

output_path = "/content/drive/MyDrive/alignement/template_double_annotation.csv"
template.to_csv(output_path, index=False, encoding='utf-8')

print(f"\n💾 Template sauvegardé : {output_path}")
print(f"\n👉 PROCHAINE ÉTAPE :")
print(f"   1. Dupliquez ce fichier une fois par annotateur (ou utilisez")
print(f"      les colonnes _annotateur1 / _annotateur2 directement)")
print(f"   2. Chaque annotateur remplit ses colonnes SANS voir celles de l'autre")
print(f"   3. Une fois les deux remplies, utilisez analyse_alignement.py")
print(f"      (Partie 1) pour calculer le Cohen's Kappa")

# Aperçu manuel avant de lancer l'annotation
print(f"\n📋 APERÇU DES {len(competences_annotation)} COMPÉTENCES SÉLECTIONNÉES")
print(f"   (relecture manuelle recommandée avant de lancer l'annotation)")
print("=" * 70)
for i, comp in enumerate(competences_annotation):
    print(f"   {i+1}. \"{comp}\"")
