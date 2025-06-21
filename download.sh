all_proxy= HF_ENDPOINT=https://hf-mirror.com ./hfd.sh wikimedia/wikipedia --dataset --include 20231101.zh/* --local-dir data/wikipedia && \
./hfd.sh openbmb/Ultra-FineWeb --dataset --include ultrafineweb-zh-*.parquet --local-dir data/Ultra-FineWeb && \
./hfd.sh openbmb/Ultra-FineWeb --dataset --include ultrafineweb-en-part-0*-of-2048.parquet --local-dir data/Ultra-FineWeb