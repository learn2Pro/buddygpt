# install miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh
# conda create -n buddygpt python=3.10
# conda activate buddygpt

export PYTHONPATH=$(pwd):$PYTHONPATH

python rlhf/dpo.py \
    --output_dir outputs/buddygpt-0.1b-dpo \
    --model_id outputs/buddygpt-0.1b-chat \
    --block_size 1024 \
    --batch_size 4 \
    --gradient_accumulation_steps 128