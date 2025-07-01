from evalscope.run import run_task
import model.modeling_tinyllm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default='outputs/buddygpt-0.1b-base')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--few_shot", type=int, default=5)
args = parser.parse_args()

task_cfg = {
    'model': args.model_id,
    'datasets': ['mmlu', 'cmmlu'],
    'eval_batch_size': args.batch_size,
    'dataset_args': {'cmmlu':{'few_shot_num': args.few_shot}},
}

run_task(task_cfg=task_cfg)