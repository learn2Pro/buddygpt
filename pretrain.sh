# install miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh
conda create -n rl python=3.10
conda activate rl

git clone git@github.com:learn2Pro/buddygpt.git

pip install -U -r requirements.txt

# cd pretrain && accelerate launch --config_file ptrain.yaml --num_processes=8 pretrain.py --batch_size=20 --gradient_accumulation_steps=64 --n_embed=1024 --n_layer=32 --ds_num_proc=300

export PYTHONPATH=$(pwd):$PYTHONPATH
export WANDB_BASE_URL=https://api.bandw.top
export HF_ENDPOINT=https://hf-mirror.com

python pretrain/pretrain.py \
    --output_dir outputs/buddygpt-0.3b-base \
    --block_size 1024 \
    --n_embed 1024 \
    --n_head 16 \
    --n_kv_head 8 \
    --n_layer 24 \
    --batch_size 20 \
    --attn_impl sdpa \
    --gradient_accumulation_steps 64