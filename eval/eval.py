from evalscope.run import run_task
import model.modeling_buddygpt


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default='outputs/buddygpt-0.2b-base-zh')
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--few_shot", type=int, default=5)
args = parser.parse_args()

task_cfg = {
    'model': args.model_id,
    'datasets': ['ceval'],
    'eval_batch_size': args.batch_size,
    'dataset_args': {'cmmlu':{'few_shot_num': args.few_shot}, 'ceval':{'few_shot_num': args.few_shot}},
}

run_task(task_cfg=task_cfg)