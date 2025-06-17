export PYTHONPATH=$(pwd):$PYTHONPATH

all_proxy= python eval/eval.py \
    --model_id learn2pro/buddygpt-0.1b-base-zh \
    --batch_size 16 \
    --few_shot 5