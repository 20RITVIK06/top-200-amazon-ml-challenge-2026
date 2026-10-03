# Amazon ML Challenge 2026 — Business Entity Resolution

**Team Heksync** (Bezawada Ritvik, Sahana A Y) — Top-200 finish

## 🏆 Headline Result

| Metric | Score |
|---|---|
| **Out-of-fold macro-F0.5 (all 2.21M training S1 entities)** | **0.9908** |
| Public leaderboard (last scored version) | 0.9882 |
| Candidate-set ceiling (perfect matcher on our blocking) | 0.9959 |

We evaluate macro-F0.5 over **every** Source-1 entity (including singletons and entities with
no candidates), using 4-fold cross-validation grouped by S1 entity. All label-derived
resources (alias tables, token log-odds) are learned fold-aware, so no validation pair ever
sees its own label.

---

## 🎯 Problem

Three-source business entity resolution: 2.21M Source-1 (S1) entities, 5.03M Source-2 and
5.29M Source-3 records. Each S2/S3 record belongs to at most one S1 entity; ~26% of records
are distractors that match nothing. The data is adversarially noised: Indic transliterations,
legal-suffix changes, leetspeak, alias markers (`fka`, `dba`), planted near-clone entities,
perturbed house numbers, and a test-only country (France).

## 🧠 Our Approach — a Cascade

```
raw TSV
  │  0. convert.py            strict tab parsing → parquet
  │  1. learn_translit.py     Indic→Latin dictionaries learned from training pairs
  │  2. prep.py + textnorm.py data-driven normalisation (3 variants)
  │  3. learn_alias.py        fold-aware locality alias tables
  ▼
Blocking (candidates.py + blocking.py)
  multi-channel TF-IDF top-K per country: name char-4-grams, address uni/bi-grams,
  combined, and a reverse channel → ~400M raw pairs
  ▼
Stage-0 pruning (stage0.py)   cheap LightGBM pre-ranker → top-10 per record, ~17M pairs (4%)
  + numkey.py                 (house number, locality, name-word) key channel
  ▼
Features (build_features.py, features.py, extra_feats.py, fmtfeats.py, rawpair.py, rawtok.py)
  ≈100 pair features: string/token/number/geography/context, learned clone detectors,
  house-number relations, raw-format provenance, acronym detection
  + adapt_unseen.py           label-free adaptation for France (unseen in train)
  ▼
Stage-1 matcher (train_oof.py, predict.py)
  LightGBM, 4-fold grouped CV, OOF evaluation
  ▼
Stage-2 collective (make_stage2_feats.py, stage2.py)
  re-scorer that sees stage-1 OOF probabilities of competing and sibling pairs
  ▼
Decision layer (expected_f.py, postproc.py, blend_nk.py)
  arg-max assignment (one S1 per record) + per-entity expected-F0.5 maximisation
  (exact Poisson-binomial DP) + OOF-selected 4-model blend
  ▼
output/matching_results.tsv + candidate_pairs.tsv
```

### Key innovations
* **Data-driven normalisation** — Indic transliteration dictionaries and locality alias
  tables are *learned from the training labels*, fold-aware, not hand-built.
* **Stage-2 collective model** — features computed from the first model's out-of-fold
  probabilities of competing and sibling pairs let the matcher see the local graph.
* **Decision-theoretic output** — arg-max assignment enforces the one-S1-per-record
  constraint; per-entity expected-F0.5 maximisation chooses how many records to emit,
  which protects singletons.
* **Unseen-country generalisation** — a shared country token plus label-free token-table
  adaptation recovered the test-only France failure mode.

## ✅ Validation Discipline

Every feature group had to pass three gates before a submission:
1. OOF macro-F0.5 improves;
2. train-vs-test distribution shift < ~0.1 SD per country;
3. per-country predicted match/empty rates on test stay unchanged.

This caught a feature group that raised OOF but *hurt* the leaderboard (record-neighbourhood
features, +0.0004 OOF → −0.011 LB) because the test set has ~40% distractors vs 26% in train.

## 🛠️ Setup

* **Python 3.12** (CPU only; no GPU, no pretrained language models, no external data)
* Hardware used: 96 CPU cores, 2 TB RAM (peak ≈ 250 GB); every step parallelises over cores.
  End-to-end wall time ≈ 3–4 h.

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

## ▶️ Run end to end

```bash
cd code/business_entity_resolution
bash src/run_all.sh <path/to/student_resource/dataset> <work_dir> <output_dir>
```

`<output_dir>` receives `matching_results.tsv` and `candidate_pairs.tsv`.
`src/run_all.sh` documents every step with exact parameters.

## 📁 Repository layout

```
README.md                        ← this file
Documentation_template.md        ← full methodology & results write-up
code/business_entity_resolution/
  requirements.txt
  README.md                      ← technical pipeline README
  src/                           ← 38 scripts, one per pipeline stage
output/
  matching_results.tsv           ← final predictions
  candidate_pairs.tsv            ← final candidate set
```

## 📊 Full results ladder

| version | change | OOF F0.5 | public LB |
|---|---|---|---|
| perfect matcher on raw blocking | ceiling | 0.99759 | – |
| v1 stage-1 | pair features | 0.98610 | – |
| v1 stage-2 | + collective features | 0.98836 | 0.97779 |
| v4 stage-2 | + clone detectors, number relations, France adaptation | 0.98867 | 0.98545 |
| a2 stage-2 | + pair-level raw-string + acronym features | 0.98977 | 0.98719 |
| t2 stage-2 | + token-level raw provenance | 0.99012 | – |
| n2 stage-2 | enlarged candidate set (+ number-key channel) | 0.99065 | 0.98821 |
| **final** | **blend n2 + t2 + tb2 + ab2** | **0.99080** | (final submission) |

See `Documentation_template.md` for the complete methodology, feature breakdown, error
analysis, and rejected approaches.
