# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Heksync
**Team Members:** Bezawada Ritvik, Sahana A Y
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We solve the three-source entity resolution task with a **cascade**: multi-channel TF-IDF
blocking (name char-4-grams, address uni/bi-grams, their combination, and a reverse channel),
a cheap LightGBM pre-ranker that prunes the ~400M blocked pairs to ~17M, a rich pairwise
LightGBM matcher (≈100 string/token/number/geography/context features), and a **second
"collective" LightGBM** that adds features computed from the first model's out-of-fold
probabilities of competing and sibling pairs. Final decisions exploit the discovered
structure of the data: each Source-2/3 record belongs to at most one Source-1 entity
(**arg-max assignment**), and per-entity **expected-F0.5 maximisation** selects which records
to output. The core pipeline uses no pretrained language model and no external data: every
dictionary (Indic transliteration, locality aliases) is learned from the provided training
pairs.

---

## 2. Methodology

### 2.1 Problem Analysis

Key findings from exploratory analysis of the training data (2.21M S1, 5.03M S2, 5.29M S3):

* **Assignment structure.** Every S2/S3 record is linked to *at most one* S1 entity
  (max multiplicity 1 over 7.64M links), and country never differs within a true pair.
  5.6% of S1 entities are singletons; ~26% of S2/S3 records match no S1 entity (distractors).
* **Names are highly duplicated.** 39% of S1 records share their normalised name with another
  S1 record (e.g. "Primary Care Group" ×253). The address is indispensable; the hardest
  negatives are *same name, different address*.
* **Name noise** (generator-style): legal-suffix add/drop/expand (Inc/Incorporated/[LLC]),
  honorific prefixes (Dr, Sri, Smt, M/s), junk prefixes (`***`, `>>`, `@`, `--`),
  leetspeak (F0undation, Char1es), accent injection (Ínc), word shuffles, generic word
  add-ons (Services, Center, Partners), `| www.x.com` suffixes, domain-only names
  (universalimpex.c0m), `(ID: 12345)` tags, and **alias markers** (a/k/a, dba, fka,
  formerly, née, t/a, trading as) whose *right-hand side* is the real name while the
  left-hand side is a generated pseudo-word ("Zetabelo fka …").
* **Transliteration.** ~25% of Indian S2 names (15% of S3) are word-by-word transliterations
  into 10 Indic scripts (Devanagari, Kannada, Telugu, Tamil, Bengali, Gujarati, …) — a small
  closed vocabulary (~1.3k words), plus native-script state names in addresses.
* **Address noise:** abbreviation (Street/St, Road/Rd), component re-ordering, `NULL`/`N/A`
  tokens, house-number formatting (`#260`, `00163`, `H.No`, `N°23`, `58BIS`, `(50)`),
  ranges and fractions (`35-39`, `19 1/2`), ordinal words (Fourth ↔ 4th), digit typos
  (789 ↔ 289), state name ↔ code, **locality substitution** by gazetteer neighbours
  (Phoenix → Maryvale, Boston → Dorchester, Hyderabad → Sangareddy), qualifiers
  (Richmond City, Mumbai Suburban), and 3.4% fully empty addresses.
* **France (test only)** follows the same generator with French vocabulary: SARL/SAS/EURL/SCI,
  `&` ↔ `ET`, street types R./BD/AV/IMP./ALL, and region ↔ département (Hauts-de-France ↔
  Nord). Every component of the pipeline is country-agnostic; country is only used to
  partition blocking and to compute per-country IDF statistics.
* **Irreducible ambiguity:** 1.4% of true links are records with an empty address *and* a
  name shared by ≥2 S1 entities; no method can resolve them. This bounds the attainable
  macro-F0.5 at roughly 0.996–0.998.

### 2.2 Solution Strategy

**Approach Type:** Blocking + cascade of gradient-boosted classifiers + graph/collective re-scoring + decision-theoretic post-processing.
**Core Innovation:** (i) data-driven normalisation (learned Indic dictionary, learned
fold-aware locality aliases), (ii) a stage-2 collective model that sees competing and sibling
evidence, and (iii) exact per-entity expected-F0.5 optimisation under the one-S1-per-record
constraint.

---

## 3. Candidate Generation (Blocking)

* **Normalisation** (`textnorm.py`): NFKC, zero-width removal, learned Indic → Latin word
  dictionary with a script transliterator fallback, accent stripping, leetspeak repair, alias
  marker splitting, domain/handle extraction, legal-form canonicalisation, honorific and
  stop-word removal for a *core name*; for addresses: component splitting, null removal,
  state/region canonicalisation per country (US codes, Indian states incl. native scripts,
  French regions/départements), house-number marker stripping, leading-zero removal,
  letter/digit splitting, ordinal words, street-type canonicalisation (EN + FR).
* **Channels** (per country, TF-IDF fitted on S1+S2+S3 of that country):
  1. `comb` — name char-4-grams (weight 0.6) ⊕ address unigrams+bigrams, top-30 S1 per record;
  2. `name` — name char-4-grams only, top-15 (records with missing / corrupted address);
  3. `addr` — address uni/bi-grams only, top-10 (records whose name was replaced);
  4. `rev` — each S1 retrieves its top-10 records on `comb` (protects crowded entities).
  Features with document frequency above a cap (20k name / 5k address) are dropped from the
  retrieval vectors, giving a ~20× speed-up with negligible recall change; exact cosines on
  the full vectors are recomputed for all retrieved pairs. Top-K uses multithreaded sparse
  matrix products (`sparse_dot_topn`).
* **Supplementary number-key channel** (`numkey.py`): truncated addresses ("17, Chennai, TN")
  defeat TF-IDF retrieval because each token alone is too common. Records and S1 entities are
  keyed by (house number, locality token, first core-name token); S1 entities sharing a key
  (keys shared by ≤ 50 entities) are ranked by name cosine and the top-2 (cosine ≥ 0.3) are added
  after pruning. This adds 1.1M train pairs containing 8,846 previously unreachable true links
  (recall 98.71% → 98.83%, candidate ceiling 0.99593 → 0.99644, final OOF +0.0004).
* **Stage-0 cascade pruning** (`stage0.py`): a LightGBM on 21 cheap features (channel
  cosines/ranks, competition gaps, empty-address flag, name frequencies), 2-fold out-of-fold
  on train, keeps the top-10 pairs per record with p0 ≥ 1e-4. The result is exactly what the
  matcher scores and what is written to `candidate_pairs.tsv`.
* **Candidate pairs generated:** train raw 407.7M → 16.6M after pruning; test raw 393.9M →
  19.3M after pruning.
* **How true matches were preserved:** raw blocking keeps 99.25% of all true train links
  (misses are almost exclusively empty-address records with ambiguous names); after pruning
  98.71%. Perfect-matcher ceiling on the final candidate set: macro-F0.5 = 0.9959.

---

## 4. Matching Model

**Features used** (`features.py`, `build_features.py`, `stage2.py`):
- *Name:* Levenshtein ratio, token-sort/token-set/partial ratios, Jaro-Winkler, full-name
  ratio, concatenated-core ratio and common-prefix (domain names), Monge-Elkan (JW) in both
  directions, first-token equality/JW, exact/sorted-token equality, web/domain-stem vs name,
  alias-part match, legal-form agreement, lengths, flags (alias / domain / Indic / web /
  handle), IDF-weighted token overlaps (Jaccard, coverage each side, missing mass), name
  frequency among S1 and all records, pseudo-name score (share of tokens never seen in S1).
- *Address:* token-set/sort/partial-set ratios, IDF-weighted overlaps, number features
  (first-number equality and edit similarity, number Jaccard, best number similarity,
  relative numeric difference, longest-number containment), state agreement, Monge-Elkan of
  alphabetic tokens, component-level best-match statistics, unmatched locality counts and
  **learned locality-alias hits** (fold-aware: out-of-fold pairs only use aliases learned from
  the other half of S1 entities), co-location counts (S1 records sharing the address).
- *Blocking / context:* channel cosines and ranks, stage-0 score, per-record and per-S1
  max-gap, rank and runner-up of each cosine, candidate counts, source (S2/S3).
- *Clone detectors (learned, fold-aware):* log-odds of each "extra" core-name token (in the
  record, not in the S1 name) and "missing" token being compatible with a true match, learned
  from training candidates (validation pairs only use statistics from the other half of the S1
  entities). The generator's planted clone entities append branch/qualifier words ("group",
  "holdings", "north", "metro", "westgate", a country word) and change the house number, while
  true-match noise appends "services", "center", "partners"; the table captures this
  (e.g. extra "group" −11.5, extra country word −9.7, extra "services" +0.3).
  Country names are mapped to one shared token (`zzcountry`) so that what is learned on
  US/India transfers to countries never seen in training.
- *House-number relations:* flags for equal, truncated (record or S1), prefix, small offset
  (≤10), single-digit substitution, same length, computed for the first and the longest S1
  number against the record numbers.
- *Stage-2 collective:* stage-1 probability, record-side competition (best other S1
  probability, margin, rank, sum), entity-side evidence (number of strong links by source,
  probability mass, max), and **sibling similarity** — TF-IDF cosine (name/address/combined)
  between the record and the records strongly linked to the same S1, and to the records of
  its best rival S1.

**Raw-format features.** Normalisation deliberately removes surface formatting, but the
generator's noise pipeline leaves traces (case changes, doubled spaces, leading zeros, '#', NULL
tokens, PO boxes, raw lengths, punctuation) whose frequencies differ between records of real
Source-1 entities and distractor records (e.g. P(matched | empty address) = 0.978 vs 0.73
overall). 16 record-level format features improve stage-1 OOF by +0.0007 while leaving the
per-country match rates on test unchanged (checked before submission).

**Pair-level raw-string features.** True-match records are produced from *this* Source-1
record by the noise pipeline, while clone/distractor records derive from another base record;
the raw (un-normalised) strings keep part of that provenance: case-insensitive exact name /
address equality, raw similarity ratios, punctuation-signature equality, raw length differences,
share and order of the address components that re-appear, and an acronym detector (record name
equals the initials, or their prefix, of the S1 name — e.g. "BM" = "Best Marketing Private
Limited", but not "Bright Energy Pvt Ltd" at the same address). +0.0008 stage-1 OOF; test match
rates per country unchanged; train/test feature shift < 0.1 SD for every feature.

**Token-level raw provenance.** Agreement of the raw token order and of per-token casing between
the S1 and record names, raw first-token and raw legal-suffix equality, token-count difference,
and equality of the raw first / last address component and component counts. +0.0005 stage-1 OOF
on top of the pair-level raw features; every feature shifts < 0.08 SD between train and test.

**Safety gates for every new feature group.** (1) OOF macro-F0.5 must improve; (2) the feature's
train-vs-test distribution (per country) must not shift by more than ~0.1 train-SD; (3) the
per-country predicted match rate and empty rate on test must stay unchanged. The rejected
neighbourhood features below failed gates (2) and (3) although they passed (1).

**Investigated and rejected: record-neighbourhood features.** We also built a sparse self-join of
every S2/S3 record's 10 nearest records and derived (i) consensus features (do a record's
near-duplicates agree with the record or with the candidate entity on house number / name) and
(ii) neighbour votes (which S1 entity the near-duplicates are matched to). They improved OOF by
a further +0.0004 but *lowered* the public score from 0.9854 to 0.9745. Diagnosis: their
distribution shifts strongly on test (e.g. the neighbour number-direction feature moves by
−0.41 train-SD for US pairs) because the test set contains many more "orphan" records (records of
entities absent from S1, which come in sibling groups sharing their own house number). On train
that pattern mostly came from exact duplicate records and correlated with true matches; on test it
marks orphans, so the model over-matched (+130K predicted pairs). Shift-aware training (dropping 18%
of train S1 entities) removed only a third of the excess, so these features were excluded.
Lesson: a label-free feature can still be unsafe if its meaning depends on the train/test
distractor mix; per-country match-rate checks on test predictions caught it.

**Unseen-country adaptation (France).** France appears only in the test set. Its name
vocabulary (e.g. generator add-ons "& Fils", "Cie", "Groupe" vs clone words "Holding",
"Participations") is unknown to the token tables. We learn it without labels: among France
candidate pairs whose names share the first token and whose street words overlap, agreement
vs disagreement of the house number is used as an address-only weak label (clones change the
number). Token log-odds from this signal are mapped to the true-label scale with an isotonic
regression fitted on train, where the same weak signal correlates 0.69 (extra tokens) / 0.84
(missing tokens) with the label-based log-odds. The adapted tables replace the token features
for France test pairs only; the models are unchanged.

**Model type:** LightGBM binary classifiers (255 leaves, lr 0.1, feature/bagging fraction
0.7, early stopping), 4-fold cross-validation grouped by S1 entity; test predictions are the
average of the 4 fold models. Stage 2 is trained on the out-of-fold stage-1 probabilities.

**Threshold selection method:** each S2/S3 record is first assigned to its arg-max S1
candidate. Then, per S1 entity, the number k of top records to output is chosen to
maximise the expected F0.5 under the calibrated probabilities (exact Poisson-binomial DP,
numba), which naturally raises the bar for adding a record to an already confident set and
protects singletons. Global-threshold selection is reported for comparison.

---

## 5. Results & Error Analysis

**Out-of-fold macro-F0.5 on all 2.21M training S1 entities** (4-fold grouped CV, arg-max
assignment + expected-F0.5 decisions), and public leaderboard where submitted:

| version | change | OOF F0.5 | public LB |
|---|---|---|---|
| perfect matcher on raw blocking candidates | ceiling | 0.99759 | – |
| perfect matcher on pruned candidates | ceiling | 0.99593 | – |
| v1 stage-1 | pair features | 0.98610 | – |
| v1 stage-2 | + collective features | 0.98836 | 0.97779 |
| v4 stage-1 | + clone detectors, number relations, country token, France adaptation, alias fold fix | 0.98650 | 0.98345 |
| v4 stage-2 | + collective features | 0.98867 | 0.98545 |
| v6 stage-3 | collective features recomputed from stage-2 | 0.98869 | – |
| f2v stage-2 | + neighbourhood-consensus + raw-format + neighbour votes (rejected, see text) | 0.98951 | 0.97447 |
| g1 stage-1 | v4 features + raw-format features | 0.98704 | – |
| g2 stage-2 | + collective features | 0.98915 | 0.98637 |
| gb2 | high-capacity g2 (lr 0.05, 511 leaves) | 0.98931 | – |
| a1 stage-1 | + pair-level raw-string + acronym features | 0.98786 | – |
| a2 stage-2 | + collective features | 0.98977 | 0.98719 |
| ab2 stage-2 | high-capacity a2 (lr 0.05, 511 leaves) | 0.98990 | – |
| t1 stage-1 | + token-level raw provenance | 0.98832 | – |
| t2 stage-2 | + collective features | 0.99012 | – |
| t2+ab2 blend | previous final | 0.99017 | 0.98786 |
| tb2 stage-2 | high-capacity t2 | 0.99024 | – |
| t2+tb2 blend | fallback | 0.99027 | – |
| n1 stage-1 | token line on the enlarged candidate set (+ number-key channel) | 0.98880 | – |
| n2 stage-2 | stage-2 on the enlarged candidate set | 0.99065 | 0.98821 |
| **final** | **blend n2 + t2 + tb2 + ab2 (new-channel pairs keep n2)** | **0.99080** | (final submission) |

* **Train → test shift.** The test set has ~40% distractor records vs 26% in train (it
  behaves like train with ~18% of S1 entities removed). Simulating this on train (dropping 18%
  of S1 entities and recomputing all competition features) costs only ≈0.001 OOF, so the large
  v1 OOF→LB gap came from France. Diagnostics per country (predicted-empty rate vs the
  generator's constant 5.6% singleton rate, matched-record share, uncertainty mass) located it:
  the planted France clones ("X France SA" at another house number) were invisible because
  country words had been stripped from core names. The fix (+0.0057 LB) generalises: a shared
  country token plus learned token log-odds.
* **Common false positives (wrong merges):** planted near-clone entities that copy an S1
  entity's name and street with a truncated / shifted house number, or append a branch word
  ("Westgate", "Metro", "North") or a country word; co-located businesses; generic chain names
  with partially missing addresses.
* **Common false negatives (missed matches):** records with an empty address whose name is
  shared by several S1 entities (irreducible: ~1.4% of true links); records whose name was
  replaced by a generated pseudo-word *and* whose address is heavily perturbed (number changed,
  locality replaced); France pairs with French-specific add-on words.

---

## 6. Conclusion

A carefully normalised, country-agnostic feature space, a pruning cascade that keeps the
matcher focused on 4% of the blocked pairs, and a collective second stage give an
out-of-fold macro-F0.5 of 0.990 against a candidate-set ceiling of 0.996 (public leaderboard 0.9872 for the last scored version). The largest
lesson was that validation on the training distribution alone hid a test-only failure mode
(France clones); per-country moment diagnostics on unlabeled test predictions found it, and a
principled, generalisable fix (shared country token + address-supervised token adaptation)
recovered most of the gap.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/`:

| file | role |
|---|---|
| `convert.py` | TSV → parquet with strict tab parsing |
| `learn_translit.py` | learns Indic → Latin dictionaries from training pairs |
| `textnorm.py`, `prep.py` | name / address normalisation (parallel) |
| `learn_alias.py` | fold-aware locality alias tables from training pairs |
| `blocking.py`, `candidates.py` | multi-channel TF-IDF blocking |
| `stage0.py` | cascade pre-ranker and pruning |
| `features.py`, `build_features.py` | pair features |
| `train_oof.py`, `predict.py` | grouped K-fold LightGBM, OOF evaluation, test inference |
| `stage2.py`, `make_stage2_feats.py` | collective features |
| `expected_f.py`, `postproc.py`, `metric.py` | decisions and metric |
| `write_submission.py` | writes `matching_results.tsv` and `candidate_pairs.tsv` |
| `run_all.sh` | end-to-end entry point |

Entry point: `bash src/run_all.sh <student_resource/dataset> <work_dir> <output_dir>`.

### B. Additional Results

* Blocking recall (true links retained): raw 99.25%, after pruning 98.71%.
* Stage-0 pruning: 407.7M → 16.6M train pairs (4.1%), 393.9M → 19.3M test pairs.
* Most informative learned clone markers (log-odds as an extra record-name token): group −11.5,
  holdings −10.5, public −10.3, exports −10.2, country word −9.7, north −9.5, metro −9.4.
* Test prediction health checks (v4): predicted-empty S1 rate France 6.0%, India 5.8%, US 5.7%
  (train singleton rate 5.6%).
