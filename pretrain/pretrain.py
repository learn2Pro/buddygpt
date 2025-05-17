from typing import Optional
import torch
import torch.nn.functional as F
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import AutoTokenizer, AutoModelForCausalLM
from model import buddygpt
from model.buddygpt import GPTConfig, BuddyGPT

output_dir = f'outputs/buddygpt-qwen3'
# THUDM/chatglm2-6b
# uer/gpt2-chinese-cluecorpussmall
# Qwen/Qwen-1_8B
# Qwen/Qwen3-0.6B
# bert-base-chinese
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
config = GPTConfig(n_block=1024, n_embed=1024, n_head=16, n_layer=16, n_vocab=len(tokenizer), n_kv_head=8)
model = BuddyGPT(config)
print(model)

def print_parameters(model):
    num_param = sum([param.numel() for param in model.parameters() if param.requires_grad])
    print(f'total param {num_param/1024/1024}m')

def sample(model, query, max_length=50):
    input_ids = tokenizer.encode(query, return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
    )
    gen_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return gen_text

print_parameters(model)

from datasets import load_dataset, concatenate_datasets
# 50m model need 20*50m = 1B token
# 100m model need 20*100m = 2B token
# 200m model need 20*200m = 4B token
# 500m model need 20*500m = 10B token

# Total tokens: 1872137976
# 1.8B token
ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train")
# 10B token * 10% = 1B token
# 10B token * 50% = 5B token
web_ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train")
# firefly ds
# 13B token
ff_ds = load_dataset("YeungNLP/firefly-pretrain-dataset", split="train")


# 拼接并切块
def group_texts_with_padding(examples):
    block_size = config.n_block
    concatenated = sum(examples["input_ids"], [])
    # result = {"input_ids": []}
    total_length = len(concatenated)
    result = {
        "input_ids": [concatenated[i:i+block_size] for i in range(0, total_length, block_size)]
    }
    return result
    
def encode(examples):
    result = tokenizer(examples['text'])
    return result

ds = ds.map(encode, batched=True, num_proc=30, remove_columns=ds.column_names)
ds = ds.map(group_texts_with_padding, batched=True, num_proc=30, remove_columns=ds.column_names)
ff_ds = ff_ds.map(encode, batched=True, num_proc=30, remove_columns=ff_ds.column_names)
ff_ds = ff_ds.map(group_texts_with_padding, batched=True, num_proc=30, remove_columns=ff_ds.column_names)
# web_ds = web_ds.map(encode, batched=True, num_proc=30, remove_columns=web_ds.column_names)
# web_ds = web_ds.map(group_texts_with_padding, batched=True, num_proc=30, remove_columns=web_ds.column_names)
ds = concatenate_datasets([ds, ff_ds])
ds

# ds['input_ids']

# Load the "all" subset or a specific subject like "computer_science"
cmmlu = load_dataset("haonan-li/cmmlu", "high_school_geography", split='dev')

# We'll use the validation set
# eval_ds = cmmlu["validation"]
def preprocess(example):
    question = example["Question"]
    choices = example["A"], example["B"], example["C"], example["D"]
    context = f"{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n答案是:"

    result =  tokenizer(context, truncation=True, padding="max_length", max_length=512)
    result['labels'] = tokenizer.encode(example['Answer'])
    return result

eval_ds = cmmlu.map(preprocess)
print(eval_ds[0])

from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # print(labels)
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

'''
accelerate launch --config_file ptrain.yaml --num_processes=1 pretrain.py
'''


@dataclass
class BuddyArguments:
    """ 模型相关参数
    """
    push_to_hub : Optional[str] = field(
        default=None, 
        metadata={"help": "if push to huggingface"}
    )

def main():
    from transformers import TrainingArguments, Trainer, TrainerCallback, DataCollatorForLanguageModeling, HfArgumentParser
    from datetime import datetime

    parser = HfArgumentParser((BuddyArguments, TrainingArguments))
    buddy_args, training_args = parser.parse_args_and_config()
    # TF32 设置（建议启用）
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.set_float32_matmul_precision("high")

    # print(sample(model, '中国首都是哪?'))
    buddygpt.FLASH = 1
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    class SampleTextCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step % 100 == 0:
                prompt = "中国首都是哪?"
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                output = model.generate(
                    input_ids=input_ids,
                    max_length=128,
                )
                gen_text = tokenizer.decode(output[0], skip_special_tokens=True)
                print(f"\n[Sample generated at step {state.global_step}]:\n{gen_text}\n")
            
            if state.global_step % 100 == 0:
                prompt = "which is the capital of china?"
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                output = model.generate(
                    input_ids=input_ids,
                    max_length=128,
                )
                gen_text = tokenizer.decode(output[0], skip_special_tokens=True)
                print(f"\n[Sample generated at step {state.global_step}]:\n{gen_text}\n")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    
    # 创建 collator
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer=tokenizer,
    #     model=model,
    #     label_pad_token_id=-100,  # 默认是 -100，loss 不计算这个 token
    #     padding=True
    # )
    
    # TL;DR
    # Action	Why
    # ✅ max_grad_norm=1.0	Clip exploding gradients
    # ✅ Lower learning_rate	Reduce gradient magnitude
    # ✅ Increase warmup_steps	Stabilize early training
    # ✅ Use gradient_accumulation_steps	Smooth out spikes
    # ✅ Monitor layers with high grad norm	Find root cause
    
    args = TrainingArguments(
        run_name=f'buddygpt-{now}',
        output_dir=output_dir,
        learning_rate=1e-4,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        logging_steps=50,
        save_steps=100,
        save_total_limit=10,
        bf16=True,
        # fp16=True,
        # max_steps=1,
        # remove_unused_columns=False,
        max_grad_norm=1.0,
        # gradient_checkpointing=True,
        gradient_accumulation_steps=128,
        eval_strategy="steps",  # or eval_strategy="steps" in newer versions
        eval_steps=500,              # Correct parameter name
        # save_safetensors=False,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        callbacks=[SampleTextCallback],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    # trainer.save_model(output_dir)
    # model.save_pretrained(f'{output_dir}/final')
    # tokenizer.save_pretrained(f'{output_dir}/final')
    if buddy_args.push_to_hub:
        trainer.push_to_hub(buddy_args.push_to_hub)

if __name__ == "__main__":
    main()