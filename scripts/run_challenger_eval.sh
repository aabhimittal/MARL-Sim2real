#!/usr/bin/env bash
# Runs a challenger model version through the Champion/Challenger promotion gate
# end-to-end. Used by CI to prove mlops/challenger.py + mlops/deploy.py are wired
# correctly -- it asserts the pipeline runs, not that any particular model quality bar is
# met (a freshly smoke-trained model isn't expected to be good).
#
# Usage: run_challenger_eval.sh [version_id]
#   With no argument, runs a fresh smoke-train first and evaluates that version
#   (the bootstrap case: no champion exists yet, so it's auto-promoted -- see
#   configs/challenger_eval.yaml's bootstrap_auto_promote).
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION_ID="${1:-}"
if [ -z "$VERSION_ID" ]; then
    echo "No version_id given -- running a fresh smoke-train first." >&2
    VERSION_ID=$(bash scripts/run_smoke_train.sh | tail -n 1)
fi

echo "Evaluating challenger: $VERSION_ID"
python3 -m marl_packing.mlops.deploy \
    --challenger "$VERSION_ID" \
    --registry-dir models/registry \
    --auto-promote
