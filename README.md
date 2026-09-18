# Alignement compétences marché-emploi / référentiel BTS SN

Code et données accompagnant l'article soumis au *Journal of Computer
Science and Technology (JCS&T)* : « Decoupled Multilingual Skill
Extraction and Semantic Alignment for Curriculum–Industry Gap
Detection ».

**Note de transparence méthodologique** : ce dépôt inclut, en plus du
pipeline final, les scripts de diagnostic et les tentatives antérieures
qui ont mené aux choix méthodologiques rapportés dans l'article
(notamment le passage d'un encodeur MiniLM à un encodeur mpnet). Chaque
script ci-dessous est explicitement étiqueté selon son statut, **vérifié
par exécution réelle** sur les données du dépôt — pas seulement par sa
description.

## Pipeline final (produit les chiffres rapportés dans l'article)

```
Offres d'emploi (texte brut)
        │
        ▼
[Étape 0, hors dépôt] Extraction NER — modèle M3, article compagnon [1]
        │
        ▼
[Étape 1] nouvel_echantillon_calibration.py
          Filtrage + tirage stratifié par décile → 151 candidats uniques
        │
        ▼
[Étape 2] template_annotation_v2_paire_fixe.py
          Calcul du top-1 par segment (encodeur utilisé à l'origine :
          MiniLM) + préparation du template à double annotation
        │
        ▼
[Étape 3, MANUELLE] Annotation par 2 personnes, en aveugle
        │
        ▼
[Étape 4] recalibration_mpnet.py  ⭐ SCRIPT CLÉ
          Reprend les MÊMES paires et labels de l'étape 2-3, mais
          recalcule les scores de similarité avec l'encodeur mpnet
          (plus robuste, cf. Détour méthodologique ci-dessous).
          Produit : κ=0,717, seuil=0,7858, exactitude=86,0%,
          spécificité=80,8%, rappel=91,7%, F1=0,8627
          → Vérifié par ré-exécution : ces chiffres correspondent
            exactement à ceux rapportés dans l'article.
        │
        ▼
[Étape 5] recalcul_final_mpnet.py
          Applique le seuil 0,7858 aux Groupes 1 (106 compétences) et
          2 (51 compétences) → résultats finaux ALIGNÉ/GAP
        │
        ▼
[Étape 6] generer_figures_article.py
          Génère les 4 figures de l'article (seuil 0,7858 codé en dur)
```

## Comparaisons de référence (Tableau 7 de l'article)

Deux scripts supplémentaires produisent les colonnes de comparaison du
Tableau 7 (baseline Jaccard, Approches 1 et 2 sans seuil) :

| Script | Rôle | Statut |
|---|---|---|
| `baseline_jaccard.py` | Similarité lexicale pure (aucun modèle) — écart moyen ≈0,28, échoue sur les paraphrases | ✅ Vérifié, chiffres cités dans l'article |
| `approche_1_et_2.py` | Approche 1 (ESCOXLM-R+DAPT, écart=0,02) et Approche 2 (Sentence-BERT MiniLM sans seuil, écart=0,26 préliminaire) | ✅ Vérifié — Approche 2 remesurée avec mpnet dans `recalibration_mpnet.py` (écart final=0,24, cf. Section 4.6) |

**Note sur le fichier source original** : ces deux scripts sont des
versions nettoyées d'un notebook original (`alignement_approche_1_et_2.py`,
~1300 lignes) contenant majoritairement du code de génération de
figures pour une version antérieure de l'article (seuil 0,55). Ce code
de figures n'est pas repris ici pour éviter toute confusion avec les
figures finales de `generer_figures_article.py`.

## Détour méthodologique (diagnostic, pas dans le pipeline numérique final)

Trois scripts documentent **pourquoi** l'encodeur mpnet a remplacé
MiniLM — conservés pour la transparence scientifique de la démarche,
même s'ils ne produisent aucun chiffre rapporté dans l'article :

| Script | Rôle |
|---|---|
| `diagnostic_top1_pare_feu.py` | Révèle qu'une paire évidente ("pare-feu") obtient un score anormalement bas avec MiniLM (0,24) |
| `diagnostic_systematique_groupe1.py` | Étend ce diagnostic aux 106 compétences du Groupe 1 |
| `test_modele_plus_puissant.py` | Compare directement MiniLM vs mpnet sur les cas problématiques identifiés, confirmant l'amélioration |

## Scripts périmés (conservés pour transparence, non utilisés dans l'article)

⚠️ **Ces deux scripts ont été testés et produisent des chiffres différents
de ceux de l'article** — ils correspondent à une itération antérieure,
basée sur l'encodeur MiniLM, avant le diagnostic ci-dessus :

| Script | Ce qu'il produit réellement (vérifié) | Chiffres de l'article |
|---|---|---|
| `calcul_seuil_final.py` | Seuil=0,8257 ; exactitude=78,0% ; spécificité=73,1% ; rappel=83,3% | Seuil=0,7858 ; exactitude=86,0% |
| `recalcul_groupes_nouveau_seuil.py` | Applique le seuil périmé 0,8257 aux Groupes 1/2 | — |

**Si vous exécutez ces deux scripts, vous n'obtiendrez pas les résultats
de l'article** — utilisez `recalibration_mpnet.py` et
`recalcul_final_mpnet.py` à la place.

## Contenu de `donnees/`

| Fichier | Contenu |
|---|---|
| `template_annotation_v2_aveugle.csv` | 50 paires, jugements bruts des 2 annotateurs (sans score) |
| `template_annotation_v2_reference.csv` | Mêmes paires, scores MiniLM (historique, étape 2) |
| `calibration_mpnet_finale.csv` | Échantillon final, scores mpnet + labels consensuels (étape 4) |
| `resultats_groupe1_MPNET_FINAL.csv` | Résultats finaux, Groupe 1 (106 compétences) |
| `resultats_groupe2_MPNET_FINAL.csv` | Résultats finaux, Groupe 2 (51 compétences) |

## Non inclus dans ce dépôt

- **Référentiel institutionnel** (`PE_SN_segments_harmonized.csv`) : non
  diffusé publiquement (restrictions d'accès institutionnelles) ;
  disponible sur demande raisonnable.
- **Offres d'emploi sources** : construites par les auteurs pour cette
  preuve de concept (Section 4.1 de l'article) ; non diffusées.
- **`entites_extraites_m3.csv`** (sortie de l'Étape 0) : dépend du modèle
  M3, décrit dans l'article compagnon [1], non redistribué ici.

## Reproduire les résultats de l'article

```bash
pip install pandas numpy scipy scikit-learn sentence-transformers matplotlib

# Étape 4 : calibration finale (mpnet) — produit κ=0,717, seuil=0,7858
python recalibration_mpnet.py

# Étape 5 : application aux Groupes 1 et 2
python recalcul_final_mpnet.py

# Étape 6 : figures
python generer_figures_article.py
```

## Citation

Si vous utilisez ce code ou ces données, merci de citer l'article
(référence complète ajoutée après publication).

## Licence

Le code de ce dépôt est publié sous licence MIT (voir le fichier
`LICENSE`). Les données (`donnees/`) sont mises à disposition pour un
usage académique/non-commercial, en cohérence avec la licence
CC BY-NC-SA 4.0 de l'article associé.
