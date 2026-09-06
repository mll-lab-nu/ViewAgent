#!/usr/bin/env bash
# Download the AI2-THOR proxy-task dataset (P2V / V2P / IVP) from HuggingFace
# and extract it into $VIEWSUITE_ROOT/data/viewagent_ai2thor/.
#
# Produces:
#   data/viewagent_ai2thor/
#     {path_to_view,view_to_path,interactive_view_planning}_{train,eval,test}.jsonl
#     FloorPlan*/...   (rendered init/option/target/top-down views)
#
# Only IVP needs the AI2-THOR render service at eval/train time; P2V/V2P read
# the pre-rendered images straight from the jsonl.
: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

python -m view_suite.utils.download_targz_hf \
    --repo=MLL-Lab/viewsuite \
    --files="viewagent_ai2thor.tar.gz" \
    --out="$VIEWSUITE_ROOT/data"
