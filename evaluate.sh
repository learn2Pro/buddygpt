export PYTHONPATH=$(pwd):$PYTHONPATH

all_proxy= python eval/eval.py \
    --model_id outputs/buddygpt-0.3b-dpo \
    --batch_size 4 \
    --few_shot 5