# Business Entity Resolution — reproducible pipeline

Cascade entity-resolution system: multi-channel TF-IDF blocking → LightGBM pre-ranker
(pruning) → pairwise LightGBM matchers → collective (stage-2) LightGBM re-scorers → OOF-selected
blend → arg-max assignment + per-entity expected-F0.5 decisions.
CPU only, no external data, no pretrained language models (all models are LightGBM, MIT licence).

## Environment

* Python 3.12, `pip install -r requirements.txt`
* Hardware used: 96 CPU cores, 2 TB RAM (peak usage ≈ 250 GB), no GPU.
  Every step parallelises over cores; end-to-end wall time ≈ 4 h.

## Run end to end

```bash
bash src/run_all.sh <path/to/student_resource/dataset> <work_dir> <output_dir>
```

`<output_dir>` receives `matching_results.tsv` and `candidate_pairs.tsv`.
`src/run_all.sh` lists every step with its exact parameters:

| step | script(s) | output |
|---|---|---|
| 0 | `convert.py` | TSV → parquet (strict tab parsing, no quoting / NA coercion) |
| 1 | `learn_translit.py` | Indic-script → Latin word dictionaries learned from training pairs |
| 2 | `prep.py` + `textnorm.py` | normalised names / addresses (three documented variants) |
| 3 | `learn_alias.py` | fold-aware locality alias tables (training labels only) |
| 4 | `candidates.py` + `blocking.py` | raw candidate pairs (TF-IDF top-K, 4 channels) |
| 5 | `stage0.py` | pruned candidate set |
| 5b | `numkey.py` | number-key channel added to the pruned set = `candidate_pairs.tsv` |
| 6 | `build_features.py` + `features.py`, `extra_feats.py` | pair features, clone detectors, number relations |
| 7 | `adapt_unseen.py` | token tables adapted to test countries unseen in train (France) |
| 8 | `fmtfeats.py`, `rawpair.py`, `rawtok.py` | record raw-format, pair-level raw-string / acronym and token-level raw provenance features |
| 9–10 | `train_oof.py`, `predict.py`, `make_stage2_feats.py` + `stage2.py` | stage-1 and stage-2 matchers (development also trained fast / high-capacity variants on three feature sets) |
| 11 | `write_submission.py`, `postproc.py`, `expected_f.py` (`blend_select.py` for blends) | final TSV files |

Analysis / experiment scripts (not needed to reproduce the submission; documented in the
methodology): `metric.py` (competition metric), `error_analysis.py` (OOF error breakdown),
`sim_shift.py`, `sa_prepare.py`, `sa_full.py`, `xfeats_apply.py` (validation under a test-like
distractor rate), `loco.py`, `loco2.py` (leave-one-country-out studies), `calibrate_unseen.py`
(per-country prior matching), `recknn.py`, `nbfeats.py`, `nbvote.py` (record-neighbourhood
features — investigated and rejected, see methodology).

## Validation protocol

All reported scores are out-of-fold macro-F0.5 over **all** 2.21M training S1 entities
(including singletons and entities without candidates), 4 folds grouped by S1 entity.
Label-derived resources (alias tables, token log-odds) are learned fold-aware so that no
validation pair ever sees its own label. Every new feature group had to pass three gates before
reaching a submission: OOF gain, no train/test distribution shift (< ~0.1 SD per country), and
unchanged per-country match rates on the test predictions.
