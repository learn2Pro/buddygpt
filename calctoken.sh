export PYTHONPATH=$(pwd):$PYTHONPATH

python ttoken.py \
    --ds data/Ultra-FineWeb \
    --field content \
    --split zh