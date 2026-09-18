# Market-Skill / BTS SN Curriculum Alignment

Code and data accompanying the article submitted to the *Journal of
Computer Science and Technology (JCS&T)*: "Decoupled Multilingual Skill
Extraction and Semantic Alignment for Curriculum–Industry Gap Detection".

**Methodological transparency note**: in addition to the final pipeline,
this repository includes the diagnostic scripts and earlier attempts
that led to the methodological choices reported in the article (notably
the switch from a MiniLM encoder to an mpnet encoder). Each script below
is explicitly labeled according to its status, **verified by actual
execution** on the repository's data — not only by its description.

## Final pipeline (produces the figures reported in the article)

```
Job postings (raw text)
        │
        ▼
[Step 0, outside this repository] NER extraction — M3 model, companion article [1]
        │
        ▼
[Step 1] nouvel_echantillon_calibration.py
         Filtering + decile-stratified sampling → 151 unique candidates
        │
        ▼
[Step 2] template_annotation_v2_paire_fixe.py
         Top-1 computation per segment (encoder originally used:
         MiniLM) + preparation of the double-annotation template
        │
        ▼
[Step 3, MANUAL] Blind annotation by 2 annotators
        │
        ▼
[Step 4] recalibration_mpnet.py  ⭐ KEY SCRIPT
         Reuses the SAME pairs and labels from steps 2-3, but
         recomputes similarity scores with the mpnet encoder
         (more robust, see Methodological detour below).
         Produces: κ=0.717, threshold=0.7858, accuracy=86.0%,
         specificity=80.8%, recall=91.7%, F1=0.8627
         → Verified by re-execution: these figures match exactly
           those reported in the article.
        │
        ▼
[Step 4bis] grid_search_validation.py
         Confirmatory check (Sections 3.4/5.3): explicit grid search over
         the same calibration sample, jointly maximizing F1 and Youden's
         J index, to verify that the empirical midpoint threshold falls
         within the grid-search-optimal decision region.
         Produces: decision region [0.7736, 0.7886] (plateau width
         0.015), which contains the midpoint threshold of 0.7858.
        │
        ▼
[Step 5] recalcul_final_mpnet.py
         Applies the 0.7858 threshold to Groups 1 (106 competencies)
         and 2 (51 competencies) → final ALIGNED/GAP results
        │
        ▼
[Step 6] generer_figures_article.py
         Generates the article's 4 figures (threshold 0.7858 hard-coded)
```

## Baseline comparisons (Table 7 of the article)

Two additional scripts produce the comparison columns of Table 7
(Jaccard baseline, Approaches 1 and 2 without threshold):

| Script | Role | Status |
|---|---|---|
| `baseline_jaccard.py` | Pure lexical similarity (no model) — average gap ≈0.28, fails on paraphrases | ✅ Verified, figures cited in the article |
| `approche_1_et_2.py` | Approach 1 (ESCOXLM-R+DAPT, gap=0.02) and Approach 2 (Sentence-BERT MiniLM without threshold, preliminary gap=0.26) | ✅ Verified — Approach 2 re-measured with mpnet in `recalibration_mpnet.py` (final gap=0.24, see Section 4.6) |

**Note on the original source file**: these two scripts are cleaned-up
versions of an original notebook (`alignement_approche_1_et_2.py`,
~1300 lines) containing mostly figure-generation code for an earlier
version of the article (threshold=0.55). That figure code is not
included here to avoid confusion with the final figures produced by
`generer_figures_article.py`.

## Methodological detour (diagnostic, not part of the final numeric pipeline)

Three scripts document **why** the mpnet encoder replaced MiniLM —
kept for scientific transparency of the process, even though they
produce no figure reported in the article:

| Script | Role |
|---|---|
| `diagnostic_top1_pare_feu.py` | Reveals that an obvious pair ("firewall") gets an abnormally low score with MiniLM (0.24) |
| `diagnostic_systematique_groupe1.py` | Extends this diagnostic to the 106 competencies of Group 1 |
| `test_modele_plus_puissant.py` | Directly compares MiniLM vs mpnet on the identified problem cases, confirming the improvement |

## Deprecated scripts (kept for transparency, not used in the article)

⚠️ **These two scripts have been tested and produce figures different
from those in the article** — they correspond to an earlier iteration,
based on the MiniLM encoder, prior to the diagnostic above:

| Script | What it actually produces (verified) | Article figures |
|---|---|---|
| `calcul_seuil_final.py` | Threshold=0.8257; accuracy=78.0%; specificity=73.1%; recall=83.3% | Threshold=0.7858; accuracy=86.0% |
| `recalcul_groupes_nouveau_seuil.py` | Applies the deprecated 0.8257 threshold to Groups 1/2 | — |

**If you run these two scripts, you will not obtain the article's
results** — use `recalibration_mpnet.py` and `recalcul_final_mpnet.py`
instead.

## Contents of `donnees/`

| File | Contents |
|---|---|
| `template_annotation_v2_aveugle.csv` | 50 pairs, raw judgments from the 2 annotators (no score) |
| `template_annotation_v2_reference.csv` | Same pairs, MiniLM scores (historical, step 2) |
| `calibration_mpnet_finale.csv` | Final sample, mpnet scores + consensus labels (step 4) |
| `resultats_groupe1_MPNET_FINAL.csv` | Final results, Group 1 (106 competencies) |
| `resultats_groupe2_MPNET_FINAL.csv` | Final results, Group 2 (51 competencies) |

## Not included in this repository

- **Institutional curriculum reference** (`PE_SN_segments_harmonized.csv`):
  not publicly released (institutional access restrictions); available
  upon reasonable request.
- **Source job postings**: compiled by the authors for this proof of
  concept (Section 4.1 of the article); not released.
- **`entites_extraites_m3.csv`** (output of Step 0): depends on the M3
  model, described in the companion article [1], not redistributed here.

## Reproducing the article's results

```bash
pip install pandas numpy scipy scikit-learn sentence-transformers matplotlib

# Step 4: final calibration (mpnet) — produces κ=0.717, threshold=0.7858
python recalibration_mpnet.py

# Step 4bis: grid-search validation of the threshold (Sections 3.4/5.3)
python grid_search_validation.py

# Step 5: application to Groups 1 and 2
python recalcul_final_mpnet.py

# Step 6: figures
python generer_figures_article.py
```

## Citation

If you use this code or data, please cite the article (full reference
to be added upon publication).

## License

The code in this repository is released under the MIT License (see the
`LICENSE` file). The data (`donnees/`) is made available for
academic/non-commercial use, consistent with the CC BY-NC-SA 4.0
license of the associated article.
