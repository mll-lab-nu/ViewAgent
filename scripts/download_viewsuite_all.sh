: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

python -m view_suite.utils.download_targz_hf \
    --repo=ViewSuite/viewsuite \
    --files=viewsuite_15k.tar.gz,mindcube.tar.gz \
    --out="$VIEWSUITE_ROOT/data"
