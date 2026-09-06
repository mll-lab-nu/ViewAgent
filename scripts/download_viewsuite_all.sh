: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

python -m view_suite.utils.download_targz_hf \
    --repo=MLL-Lab/viewsuite \
    --files="viewagent15k_scannet_open3d.tar.gz,viewagent15k_scannet_gs_test.tar.gz,viewagent_mindcube.tar.gz" \
    --out="$VIEWSUITE_ROOT/data"
