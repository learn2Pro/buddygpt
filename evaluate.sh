export PYTHONPATH=$(pwd):$PYTHONPATH

all_proxy= python eval/eval.py \
    --model_id outputs/buddygpt-0.1b-chat-maxode \
    --batch_size 8 \
    --few_shot 5