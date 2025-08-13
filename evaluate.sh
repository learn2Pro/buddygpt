export PYTHONPATH=$(pwd):$PYTHONPATH

all_proxy= python eval/eval.py \
    --model_id outputs/buddygpt-0.3b-cpt \
    --batch_size 4 \
    --few_shot 5


all_proxy= python eval/eval.py \
    --model_id outputs/buddygpt-1b-moe-mla-base \
    --batch_size 4 \
    --few_shot 5