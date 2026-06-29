: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

hf download MLL-Lab/viewagent-all-qwen25vl7b \
  --local-dir "$VIEWSUITE_ROOT/model/qwen25-ivp/viewagent-all-qwen25vl7b"

hf download MLL-Lab/viewagent-ivp-qwen25vl7b \
  --local-dir "$VIEWSUITE_ROOT/model/qwen25-ivp/viewagent-ivp-qwen25vl7b"
