# install miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh
# conda create -n buddygpt python=3.10
# conda activate buddygpt

export PYTHONPATH=$(pwd):$PYTHONPATH
export WANDB_BASE_URL=https://api.bandw.top
export HF_ENDPOINT=https://hf-mirror.com

python sft/sft.py \
    --output_dir outputs/buddygpt-0.3b-chat \
    --model_id outputs/buddygpt-0.3b-base/checkpoint-38500 \
    --block_size 4196 \
    --batch_size 2 \
    --gradient_accumulation_steps 256 \
    --ds_batch_size 8192