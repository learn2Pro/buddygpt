from evalscope.run import run_task
import model.modeling_tinyllm

task_cfg = {
    'model': 'outputs/buddygpt-0.2b-chat-zh/checkpoint-1200',
    'datasets': ['cmmlu'],
    'eval_batch_size': 16,
}

run_task(task_cfg=task_cfg)