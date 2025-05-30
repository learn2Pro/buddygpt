from evalscope.run import run_task
import modeling_tinyllm

task_cfg = {
    'model': 'qwen/qwen3-0.6b',
    'datasets': ['cmmlu'],
    'eval_batch_size': 16,
}

run_task(task_cfg=task_cfg)