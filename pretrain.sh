# install miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh
conda create -n buddygpt python=3.10
conda activate buddygpt

pip install -U -r requirements.txt

export PYTHONPATH=$(pwd):$PYTHONPATH

python pretrain/pretrain.py \
    --output_dir outputs/buddygpt-0.4b-base \
    --block_size 1024 \
    --n_embed 1024 \
    --n_head 16 \
    --n_layer 24 \
    --batch_size 20 \
    --attn_impl sdpa \
    --gradient_accumulation_steps 64