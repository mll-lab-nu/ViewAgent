: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

hf download MLL-Lab/viewsuite-all-qwen25vl7b \
  --local-dir "$VIEWSUITE_ROOT/model/qwen25-ivp/viewsuite-all-qwen25vl7b"

hf download MLL-Lab/viewsuite-ivp-qwen25vl7b \
  --local-dir "$VIEWSUITE_ROOT/model/qwen25-ivp/viewsuite-ivp-qwen25vl7b"
