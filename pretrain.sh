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
    --output_dir outputs/buddygpt-0.4b-moe-base \
    --block_size 1024 \
    --n_embed 1024 \
    --n_head 16 \
    --n_kv_head 8 \
    --n_layer 24 \
    --batch_size 4 \
    --attn_impl sdpa \
    --gradient_accumulation_steps 512

python pretrain/pretrain.py \
    --output_dir outputs/buddygpt-0.7b-moe-mla-base \
    --block_size 1024 \
    --n_embed 1536 \
    --n_head 16 \
    --n_layer 24 \
    --attn_impl mla \
    --q_lora_rank 16 \
    --qk_rope_head_dim 24 \
    --qk_nope_head_dim 72 \
    --kv_lora_rank 16 \
    --v_head_dim 96 \
    --n_expert 12 \
    --n_expert_per_token 2 \
    --n_group 2 \
    --n_topk_group 1 \
    --moe_intermediate_size 256 \
    --batch_size 2 \
    --gradient_accumulation_steps 1024

# python pretrain/pretrain.py \
#     --output_dir outputs/buddygpt-0.3b-cpt \
#     --cpt_path learn2pro/buddygpt-0.3b-base \
#     --block_size 1024 \
#     --n_embed 1024 \
#     --n_head 16 \
#     --n_kv_head 8 \
#     --n_layer 24 \
#     --batch_size 4 \
#     --attn_impl sdpa \
#     --gradient_accumulation_steps 512