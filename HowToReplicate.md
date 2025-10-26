
# Reproduction Runbook

This document provides step-by-step instructions to reproduce the newcomer outcomes and toxicity analysis pipeline using the provided scripts:

# Small note:
If you wanted Unzip the provided zips into a folder named mined/ at the project root. After extraction, the directory should look like this (paths that matter):

project-root/
├─ mined/
│  ├─ shard_01.raw.json
│  ├─ shard_01.features.json
│  ├─ shard_01.scored.json
│  ├─ ...
│  ├─ shard_20.raw.json
│  ├─ shard_20.features.json
│  ├─ shard_20.scored.json
│  ├─ comments_365d/
│  │  ├─ shard_01/**/tox_index.json
│  │  ├─ ... (and there will be repo level comments inside of shard_xx/repo
│  │  └─ shard_20/**/tox_index.json ...


```
rac_mine.py
tox_score_comments.py
analyze_rac.py
rac_build.py  (only if re-collecting repos, not required for reproduction)
```

The canonical input repository file (`repos_TIMESTAMP.json`) is provided.
All commands assume execution from the **project root directory**.

---

## 1. Dependencies

### Python 3.10 or newer

Install the following Python packages inside a virtual environment:

```
requests
python-dateutil
PyGithub
numpy
pandas
matplotlib
statsmodels
scipy
transformers
torch
tqdm
```

### Additional CLI requirement (macOS only)

`jq` must be installed for merge operations.

---

## 2. Environment Variables

Set before running the pipeline.

macOS:

```bash
export GITHUB_TOKEN="<YOUR_TOKEN>"
export RAC_ASOF_UTC="2025-09-01T00:00:00Z"
export RAC_WIN_START_UTC="2024-09-01T00:00:00Z"
```

Windows PowerShell:

```powershell
$env:GITHUB_TOKEN      = "<YOUR_TOKEN>"
$env:RAC_ASOF_UTC      = "2025-09-01T00:00:00Z"
$env:RAC_WIN_START_UTC = "2024-09-01T00:00:00Z"
```

---

## 3. Virtual Environment Setup

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install requests python-dateutil PyGithub numpy pandas matplotlib statsmodels scipy transformers torch tqdm
brew install jq
```

### Windows PowerShell

```powershell
py -m venv .venv
. .\.venv\Scripts\Activate
python -m pip install --upgrade pip
python -m pip install requests python-dateutil PyGithub numpy pandas matplotlib statsmodels scipy transformers torch tqdm
# Optional if you want jq:
# winget install jqlang.jq
```

---

## 4. Prepare Canonical Working Set (100 repos)

macOS:

```bash
jq '.[0:100]' repos_TIMESTAMP.json > repos_100.json
```

Windows (requires jq):

```powershell
jq '.[0:100]' repos_TIMESTAMP.json > repos_100.json
```

---

## 5. Create 20 Shards (5 repos each)

### macOS

```bash
mkdir -p shards
for i in $(seq 0 19); do
  jq --argjson i "$i" '.[($i*5):(($i+1)*5)]' repos_100.json \
    > "shards/shard_$(printf "%02d" $((i+1))).json"
done
```

### Windows PowerShell (requires jq)

```powershell
mkdir shards | Out-Null
foreach ($i in 0..19) {
  jq --argjson i $i '.[($i*5):(($i+1)*5)]' repos_100.json `
    > ("shards/shard_{0:d2}.json" -f ($i+1))
}
```

---

## 6. Pipeline Per Shard

Replace `XX` with `01` to `20`.

### macOS

```bash
python rac_mine.py \
  --log logs/shard_XX.mine.log \
  mine_from_json \
  --json shards/shard_XX.json \
  --out  mined/shard_XX.raw.json \
  --sleep 0.2

python rac_mine.py \
  --log logs/shard_XX.score.log \
  score \
  --inp mined/shard_XX.raw.json \
  --out mined/shard_XX.scored.json

python rac_mine.py \
  --log logs/shard_XX.features.log \
  features \
  --inp  mined/shard_XX.scored.json \
  --out  mined/shard_XX.features.json \
  --dump_comments_dir mined/comments_365d \
  --shard_id shard_XX \
  --sleep 0.2
```

### Windows

```powershell
python rac_mine.py `
  --log logs/shard_XX.mine.log `
  mine_from_json `
  --json shards/shard_XX.json `
  --out  mined/shard_XX.raw.json `
  --sleep 0.2

python rac_mine.py `
  --log logs/shard_XX.score.log `
  score `
  --inp mined/shard_XX.raw.json `
  --out mined/shard_XX.scored.json

python rac_mine.py `
  --log logs/shard_XX.features.log `
  features `
  --inp  mined/shard_XX.scored.json `
  --out  mined/shard_XX.features.json `
  --dump_comments_dir mined/comments_365d `
  --shard_id shard_XX `
  --sleep 0.2
```

---

## 7. Toxicity Scoring

Model:

```
SkolkovoInstitute/roberta_toxicity_classifier
```

### macOS

```bash
for d in mined/comments_365d/shard_*; do
  [ -d "$d" ] || continue
  sid=$(basename "$d")
  python tox_score_comments.py \
    --dump_root mined/comments_365d \
    --shard_id "$sid" \
    --model SkolkovoInstitute/roberta_toxicity_classifier \
    --threshold 0.5 \
    --batch 16 \
    --device cpu
done
```

### Windows

```powershell
Get-ChildItem -Directory mined/comments_365d/shard_* | ForEach-Object {
  $sid = $_.Name
  python tox_score_comments.py `
    --dump_root mined/comments_365d `
    --shard_id $sid `
    --model SkolkovoInstitute/roberta_toxicity_classifier `
    --threshold 0.5 `
    --batch 16 `
    --device cpu
}
```

---

## 8. Merge to Unified Files

### macOS (jq)

```bash
jq -s 'add' ./mined/shard_*.features.json \
  > ./mined/all.features.json

jq -s 'add' ./mined/shard_*.scored.json \
  > ./mined/all.scored.json

jq -s '.' ./mined/comments_365d/shard_*/**/tox_index.json \
  > ./mined/all.tox_index.json
```

### Windows PowerShell (no jq required)

```powershell
# Features
$features = Get-ChildItem mined -Filter "shard_*.features.json" | Sort-Object Name |
  ForEach-Object { Get-Content $_ -Raw | ConvertFrom-Json }
$featuresFlat = @(); foreach ($arr in $features) { $featuresFlat += $arr }
$featuresFlat | ConvertTo-Json -Depth 100 > mined/all.features.json

# Scored
$scored = Get-ChildItem mined -Filter "shard_*.scored.json" | Sort-Object Name |
  ForEach-Object { Get-Content $_ -Raw | ConvertFrom-Json }
$scoredFlat = @(); foreach ($arr in $scored) { $scoredFlat += $arr }
$scoredFlat | ConvertTo-Json -Depth 100 > mined/all.scored.json

# Toxicity
$tox = Get-ChildItem mined/comments_365d -Recurse -Filter "tox_index.json" |
  ForEach-Object { Get-Content $_ -Raw | ConvertFrom-Json }
$tox | ConvertTo-Json -Depth 100 > mined/all.tox_index.json
```

---

## 9. Final Statistical Analysis

```bash
python analyze_rac.py \
  --features_json mined/all.features.json \
  --scored_json   mined/all.scored.json \
  --tox_json      mined/all.tox_index.json \
  --min_pr_denom  5 \
  --alpha         0.05 \
  --aggregate_retention \
  --outdir analysis_out_aggregated
```

Outputs are placed in:

```
analysis_out_aggregated/
```

This directory contains:

* CSV files with statistics
* Plots for retention, merge rate, conversion, and toxicity impact

---

## 10. Optional: Run All Shards Automatically

### macOS

```bash
for n in $(seq -w 01 20); do
  python rac_mine.py --log logs/shard_${n}.mine.log mine_from_json --json shards/shard_${n}.json --out mined/shard_${n}.raw.json --sleep 0.2
  python rac_mine.py --log logs/shard_${n}.score.log score --inp mined/shard_${n}.raw.json --out mined/shard_${n}.scored.json
  python rac_mine.py --log logs/shard_${n}.features.log features --inp mined/shard_${n}.scored.json --out mined/shard_${n}.features.json --dump_comments_dir mined/comments_365d --shard_id shard_${n} --sleep 0.2
done
```

### Windows PowerShell

```powershell
$nums = 1..20 | ForEach-Object { "{0:d2}" -f $_ }
foreach ($n in $nums) {
  python rac_mine.py --log "logs/shard_$n.mine.log"    mine_from_json --json "shards/shard_$n.json" --out "mined/shard_$n.raw.json" --sleep 0.2
  python rac_mine.py --log "logs/shard_$n.score.log"   score         --inp "mined/shard_$n.raw.json"    --out "mined/shard_$n.scored.json"
  python rac_mine.py --log "logs/shard_$n.features.log" features     --inp "mined/shard_$n.scored.json" --out "mined/shard_$n.features.json" --dump_comments_dir "mined/comments_365d" --shard_id "shard_$n" --sleep 0.2
}
```

---

## Completion

At this point the full pipeline is executed, and results are ready for interpretation in `analysis_out_aggregated/`.

---
