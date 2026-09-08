"""
Génération des figures pour l'article (version CAEE)
=====================================================================

Produit les figures en PNG haute résolution à partir de vos fichiers
CSV finaux, prêtes à insérer dans le document Word.

Fichiers nécessaires :
  - calibration_mpnet_finale.csv
  - resultats_groupe1_MPNET_FINAL.csv
  - resultats_groupe2_MPNET_FINAL.csv

Sortie : figure1_distribution_seuil.png, figure2_matrice_confusion.png,
         figure3_resultats_groupes.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SEUIL = 0.7858
plt.rcParams['font.size'] = 11
plt.rcParams['savefig.dpi'] = 300


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
# FIGURE 1 : Distribution des similarités MATCH/NON_MATCH + seuil
# =========================================================================

df_calib = lire_csv_robuste(
    "/content/drive/MyDrive/alignement/calibration_mpnet_finale.csv"
)

scores_match = df_calib[df_calib['type_pair_final'] == 'MATCH']['similarite_mpnet']
scores_non_match = df_calib[df_calib['type_pair_final'] == 'NON_MATCH']['similarite_mpnet']

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 1, 25)
ax.hist(scores_non_match, bins=bins, alpha=0.6, label='NON_MATCH', color='#E07B54')
ax.hist(scores_match, bins=bins, alpha=0.6, label='MATCH', color='#4C8C7D')
ax.axvline(SEUIL, color='black', linestyle='--', linewidth=1.5,
           label=f'Seuil = {SEUIL}')
ax.set_xlabel('Score de similarité cosinus')
ax.set_ylabel("Nombre de paires")
ax.set_title("Distribution des similarités MATCH / NON_MATCH\n(échantillon de calibration, n=50)")
ax.legend()
fig.tight_layout()
fig.savefig('/content/drive/MyDrive/alignement/figure1_distribution_seuil.png')
print("✅ Figure 1 sauvegardée : figure1_distribution_seuil.png")
plt.close(fig)

# =========================================================================
# FIGURE 2 : Matrice de confusion + métriques
# =========================================================================

vp = int((scores_match >= SEUIL).sum())
fn = int((scores_match < SEUIL).sum())
vn = int((scores_non_match < SEUIL).sum())
fp = int((scores_non_match >= SEUIL).sum())

rappel = vp / (vp + fn)
specificite = vn / (vn + fp)
exactitude = (vp + vn) / (vp + fn + vn + fp)
precision = vp / (vp + fp)
f1 = 2 * precision * rappel / (precision + rappel)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Matrice de confusion
matrice = np.array([[vp, fn], [fp, vn]])
im = axes[0].imshow(matrice, cmap='Blues')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Prédit MATCH', 'Prédit NON_MATCH'])
axes[0].set_yticklabels(['Réel MATCH', 'Réel NON_MATCH'])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(matrice[i, j]), ha='center', va='center',
                      fontsize=16, fontweight='bold',
                      color='white' if matrice[i, j] > matrice.max()/2 else 'black')
axes[0].set_title("Matrice de confusion (n=50)")

# Barres de métriques
metriques = {'Rappel': rappel, 'Spécificité': specificite,
             'Exactitude': exactitude, 'Précision': precision, 'F1': f1}
axes[1].bar(metriques.keys(), metriques.values(), color='#4C8C7D')
axes[1].set_ylim(0, 1)
axes[1].set_title("Performance au seuil calibré")
for i, (k, v) in enumerate(metriques.items()):
    axes[1].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
axes[1].tick_params(axis='x', rotation=30)

fig.tight_layout()
fig.savefig('/content/drive/MyDrive/alignement/figure2_matrice_confusion.png')
print("✅ Figure 2 sauvegardée : figure2_matrice_confusion.png")
plt.close(fig)

# =========================================================================
# FIGURE 3 : Résultats Groupe 1 et Groupe 2 (alignés vs GAPS)
# =========================================================================

df_g1 = lire_csv_robuste(
    "/content/drive/MyDrive/alignement/resultats_groupe1_MPNET_FINAL.csv"
)
df_g2 = lire_csv_robuste(
    "/content/drive/MyDrive/alignement/resultats_groupe2_MPNET_FINAL.csv"
)

n_g1_aligne = (df_g1['statut'] == 'ALIGNE').sum()
n_g1_gap = (df_g1['statut'] == 'GAP').sum()
n_g2_aligne = (df_g2['statut'] == 'ALIGNE').sum()
n_g2_gap = (df_g2['statut'] == 'GAP').sum()

fig, ax = plt.subplots(figsize=(7, 5))
groupes = ['Groupe 1\n(8 offres, n=106)', 'Groupe 2\n(2 offres test, n=51)']
alignes = [n_g1_aligne, n_g2_aligne]
gaps = [n_g1_gap, n_g2_gap]

x = np.arange(len(groupes))
width = 0.5
ax.bar(x, alignes, width, label='ALIGNÉ', color='#4C8C7D')
ax.bar(x, gaps, width, bottom=alignes, label='GAP', color='#E07B54')

for i in range(len(groupes)):
    total = alignes[i] + gaps[i]
    ax.text(i, alignes[i]/2, f"{alignes[i]}\n({alignes[i]/total*100:.1f}%)",
            ha='center', va='center', fontweight='bold', color='white')
    ax.text(i, alignes[i] + gaps[i]/2, f"{gaps[i]}\n({gaps[i]/total*100:.1f}%)",
            ha='center', va='center', fontweight='bold', color='white')

ax.set_xticks(x)
ax.set_xticklabels(groupes)
ax.set_ylabel("Nombre de compétences")
ax.set_title(f"Résultats d'alignement (seuil = {SEUIL})")
ax.legend()
fig.tight_layout()
fig.savefig('/content/drive/MyDrive/alignement/figure3_resultats_groupes.png')
print("✅ Figure 3 sauvegardée : figure3_resultats_groupes.png")
plt.close(fig)

print("\n" + "=" * 70)
print("Toutes les figures sont dans votre dossier Drive 'alignement/'.")
print("Téléchargez-les et insérez-les dans le document Word aux")
print("emplacements marqués [FIGURE X : ... à régénérer].")
print("=" * 70)
