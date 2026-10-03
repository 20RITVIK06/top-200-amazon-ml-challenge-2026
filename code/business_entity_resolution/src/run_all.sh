#!/usr/bin/env bash
# End-to-end reproduction: raw TSV -> blocking -> matching -> output/{matching_results,candidate_pairs}.tsv
#
# usage: bash src/run_all.sh <path/to/student_resource/dataset> <work_dir> <output_dir>
# Hardware used: 96 CPU cores (Xeon, AVX-512), 2 TB RAM (peak ~250 GB), no GPU. Wall time ~3 h.
set -euo pipefail
DATA=$(realpath "${1:?dataset dir}")
WORK=$(realpath -m "${2:?work dir}")
OUT=$(realpath -m "${3:?output dir}")
SRC=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
mkdir -p "$WORK"/{data,res,norm_block,norm_s0,norm,cand,feats,models} "$OUT"
cd "$SRC"

# 0. TSV -> parquet (strict tab parsing)
$PY convert.py "$DATA" "$WORK/data"

# 1. Indic-script -> Latin dictionaries learned from labelled training pairs
$PY learn_translit.py "$WORK/data" "$WORK/res/translit.json"

# 2. normalisation. prep.py args: n_jobs drop_locality country_token
#    norm_block: representation used for blocking; norm_s0: for stage-0 pruning; norm: matcher features
for S in train test; do
  $PY prep.py "$WORK/data" "$WORK/res/translit.json" $S "$WORK/norm_block" 94 0 0
  $PY prep.py "$WORK/data" "$WORK/res/translit.json" $S "$WORK/norm_s0"    94 1 0
  $PY prep.py "$WORK/data" "$WORK/res/translit.json" $S "$WORK/norm"       94 1 1
done

# 3. locality alias tables (fold-aware; learned from training labels only)
$PY learn_alias.py "$WORK/norm" "$WORK/data/train_gt.parquet" "$WORK/res/alias.json" 3

# 4. blocking: multi-channel TF-IDF top-K (comb / name / addr / reverse)
$PY candidates.py "$WORK/norm_block" train "$WORK/cand"
$PY candidates.py "$WORK/norm_block" test  "$WORK/cand"

# 5. stage-0 cascade pruning (top-10 per record, p0 >= 1e-4) -> the matcher's candidate set
$PY stage0.py train "$WORK/norm_s0" "$WORK/cand" "$WORK/data/train_gt.parquet" 10 1e-4
$PY stage0.py test  "$WORK/norm_s0" "$WORK/cand" 10 1e-4

# 5b. supplementary blocking channel: (house number, locality, first name word) keys ranked by
#     name similarity -> recovers truncated-address records; union = final candidate set
$PY numkey.py "$WORK/norm_block" "$WORK/cand" train "$WORK/cand_nk" 2 0.3 50
$PY numkey.py "$WORK/norm_block" "$WORK/cand" test  "$WORK/cand_nk" 2 0.3 50
CAND="$WORK/cand_nk"

# 6. pair features + token log-odds / house-number relation features
$PY build_features.py "$WORK/norm" "$CAND" train "$WORK/feats" "$WORK/res/alias.json"
$PY build_features.py "$WORK/norm" "$CAND" test  "$WORK/feats" "$WORK/res/alias.json"
$PY extra_feats.py "$WORK/norm" "$CAND" "$WORK/data" "$WORK/feats"

# 7. unsupervised adaptation of token tables to test countries unseen in train (France)
$PY adapt_unseen.py "$WORK/norm" "$CAND" "$WORK/data" "$WORK/feats"

# 8. record raw-format, pair-level raw-string / acronym and token-level raw provenance features
for S in train test; do
  $PY fmtfeats.py "$WORK/data" "$CAND" $S "$WORK/feats/${S}_fmtfeats.parquet"
  $PY rawpair.py  "$WORK/data" "$CAND" $S "$WORK/feats/${S}_rawpair2.parquet"
  $PY rawtok.py   "$WORK/data" "$CAND" $S "$WORK/feats/${S}_rawtok.parquet"
done
XTR="$WORK/feats/train_xfeats.parquet $WORK/feats/train_fmtfeats.parquet $WORK/feats/train_rawpair2.parquet $WORK/feats/train_rawtok.parquet"
XTE="$WORK/feats/test_xfeats_adapted.parquet $WORK/feats/test_fmtfeats.parquet $WORK/feats/test_rawpair2.parquet $WORK/feats/test_rawtok.parquet"
$PY make_stage2_feats.py prepare "$WORK/norm" "$CAND" train
$PY make_stage2_feats.py prepare "$WORK/norm" "$CAND" test

# 9. stage 1: 4-fold grouped LightGBM (OOF on train, fold-average on test)
$PY train_oof.py "$CAND/train_cand.parquet" "$WORK/feats/train_feats.parquet" "$WORK/data/train_gt.parquet" \
    "$WORK/data/train_s1.parquet" "$WORK/models/n1" --folds 4 --rounds 3000 --lr 0.1 --extra $XTR
$PY predict.py "$WORK/models/n1" 4 "$WORK/feats/test_feats.parquet" "$WORK/models/n1_test.npy" --extra $XTE

# 10. stage 2: same features + collective features from the stage-1 probabilities
$PY make_stage2_feats.py "$WORK/norm" "$CAND" train "$WORK/models/n1_oof.npy" "$WORK/feats/train_n2c.parquet"
$PY make_stage2_feats.py "$WORK/norm" "$CAND" test  "$WORK/models/n1_test.npy" "$WORK/feats/test_n2c.parquet"
$PY train_oof.py "$CAND/train_cand.parquet" "$WORK/feats/train_feats.parquet" "$WORK/data/train_gt.parquet" \
    "$WORK/data/train_s1.parquet" "$WORK/models/n2" --folds 4 --rounds 3000 --lr 0.1 --extra $XTR "$WORK/feats/train_n2c.parquet"
$PY predict.py "$WORK/models/n2" 4 "$WORK/feats/test_feats.parquet" "$WORK/models/n2_test.npy" \
    --extra $XTE "$WORK/feats/test_n2c.parquet"

# 11. companion models on the original (pre-channel) candidate set, same feature groups
#     (t2: token line fast, tb2: token line high-capacity, ab2: raw-pair line high-capacity)
BIG="--rounds 5000 --lr 0.05 --leaves 511 --min_leaf 50 --ff 0.6"
for S in train test; do
  $PY build_features.py "$WORK/norm" "$WORK/cand" $S "$WORK/feats_o" "$WORK/res/alias.json"
  $PY fmtfeats.py "$WORK/data" "$WORK/cand" $S "$WORK/feats_o/${S}_fmtfeats.parquet"
  $PY rawpair.py  "$WORK/data" "$WORK/cand" $S "$WORK/feats_o/${S}_rawpair2.parquet"
  $PY rawtok.py   "$WORK/data" "$WORK/cand" $S "$WORK/feats_o/${S}_rawtok.parquet"
done
$PY extra_feats.py "$WORK/norm" "$WORK/cand" "$WORK/data" "$WORK/feats_o"
$PY adapt_unseen.py "$WORK/norm" "$WORK/cand" "$WORK/data" "$WORK/feats_o"
$PY make_stage2_feats.py prepare "$WORK/norm" "$WORK/cand" train
$PY make_stage2_feats.py prepare "$WORK/norm" "$WORK/cand" test
oline () {  # $1 stage-1 prefix, $2 stage-2 prefix, $3 feature set a|t, $4.. train_oof args
  local P1=$1 P2=$2 SET=$3; shift 3
  local O="$WORK/feats_o"
  local XT="$O/train_xfeats.parquet $O/train_fmtfeats.parquet $O/train_rawpair2.parquet"
  local XE="$O/test_xfeats_adapted.parquet $O/test_fmtfeats.parquet $O/test_rawpair2.parquet"
  if [ "$SET" = t ]; then XT="$XT $O/train_rawtok.parquet"; XE="$XE $O/test_rawtok.parquet"; fi
  $PY train_oof.py "$WORK/cand/train_cand.parquet" "$O/train_feats.parquet" "$WORK/data/train_gt.parquet" \
      "$WORK/data/train_s1.parquet" "$WORK/models/$P1" --folds 4 "$@" --extra $XT
  $PY predict.py "$WORK/models/$P1" 4 "$O/test_feats.parquet" "$WORK/models/${P1}_test.npy" --extra $XE
  $PY make_stage2_feats.py "$WORK/norm" "$WORK/cand" train "$WORK/models/${P1}_oof.npy" "$O/train_${P2}c.parquet"
  $PY make_stage2_feats.py "$WORK/norm" "$WORK/cand" test "$WORK/models/${P1}_test.npy" "$O/test_${P2}c.parquet"
  $PY train_oof.py "$WORK/cand/train_cand.parquet" "$O/train_feats.parquet" "$WORK/data/train_gt.parquet" \
      "$WORK/data/train_s1.parquet" "$WORK/models/$P2" --folds 4 "$@" --extra $XT "$O/train_${P2}c.parquet"
  $PY predict.py "$WORK/models/$P2" 4 "$O/test_feats.parquet" "$WORK/models/${P2}_test.npy" --extra $XE "$O/test_${P2}c.parquet"
}
oline t1 t2 t --rounds 3000 --lr 0.1
oline tb1 tb2 t $BIG
oline ab1 ab2 a $BIG

# 12. final: OOF-selected blend (n2 + t2 + tb2 + ab2, OOF macro-F0.5 0.99080); pairs added by the
#     number-key channel keep n2's probability; arg-max assignment + expected-F0.5 -> output files
$PY blend_nk.py "$WORK" "$OUT"
