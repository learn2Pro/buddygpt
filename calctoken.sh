export PYTHONPATH=$(pwd):$PYTHONPATH

python ttoken.py \
    --ds openbmb/Ultra-FineWeb \
    --field content \
    --split zh