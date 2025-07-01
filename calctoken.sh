export PYTHONPATH=$(pwd):$PYTHONPATH

python ttoken.py \
    --ds data/Ultra-FineWeb \
    --ds_batch_size 2048 \
    --field content \
    --split zh

python ttoken.py \
    --ds data/Ultra-FineWeb \
    --ds_batch_size 2048 \
    --field prompt \
    --field1 response \
    --split zh