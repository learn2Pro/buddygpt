from transformers import AutoTokenizer
from configuration_buddygpt import BuddyGPTConfig
from modeling_buddygpt import BuddyGPTForCausalLM
import torch
import torch.nn.functional as F
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

output_dir = f'outputs/buddygpt-qwen3'
block_size = 1024
# uer/gpt2-chinese-cluecorpussmall
# Qwen/Qwen3-0.6B
# gpt2
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B' ,trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
config = BuddyGPTConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=2048,
        rope_theta=10000.0,
        num_seq_len=block_size,
        vocab_size=len(tokenizer),
        tie_word_embeddings=True,
    ) 
model = BuddyGPTForCausalLM(config)
print(tokenizer.pad_token, tokenizer.pad_token_id)
print(tokenizer.eos_token, tokenizer.eos_token_id)
print(model)


def print_parameters(model):
    num_param = sum([param.numel() for param in model.parameters() if param.requires_grad])
    print(f'total param {num_param/1024/1024}m')

def sample(model, query, max_length=50):
    input_ids = tokenizer.encode(query, return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_length,
    )
    gen_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return gen_text

model.to(device)
print_parameters(model)
sample(model, '中国首都是哪?')
# print(sum([p.numel() for p in model.parameters()]) / 1024 / 1024)


## load dataset

from datasets import load_dataset, concatenate_datasets
# 50m model need 20*50m = 1B token
# 100m model need 20*100m = 2B token
# 200m model need 20*200m = 4B token
# 500m model need 20*500m = 10B token

# Total tokens: 1872137976
# 1.8B token
ds = load_dataset("pleisto/wikipedia-cn-20230720-filtered", split="train")
# zhihu instruction
zhihu_ds = load_dataset("wangrui6/Zhihu-KOL", split="train")
# 10B token
web_ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train")
# firefly ds
# 13B token
ff_ds = load_dataset("YeungNLP/firefly-pretrain-dataset", split="train")
# novel ds
novel_ds = load_dataset("wdndev/webnovel-chinese", split="train")

# 拼接并切块
def group_texts_with_padding(examples):
    # block_size = block_size
    concatenated = sum(examples["input_ids"], [])
    # result = {"input_ids": []}
    total_length = len(concatenated)
    result = {
        "input_ids": [concatenated[i:i+block_size] for i in range(0, total_length, block_size)]
    }
    return result

def encode(examples, field: str = 'text'):
    result = tokenizer(examples[field])
    return result

def encode_instruction(examples):
    input_ids = []
    attention_mask = []
    for i in range(len(examples['INSTRUCTION'])):
        instruction = examples['INSTRUCTION'][i]
        response = examples['RESPONSE'][i]
        build_instruction = f"### Instruction:\n{instruction}\n### Response:\n{response}"
        tokenized_instruction = tokenizer(build_instruction)
        input_ids.append(tokenized_instruction['input_ids'])
        attention_mask.append(tokenized_instruction['attention_mask'])
    result = {}
    result['input_ids'] = input_ids
    result['attention_mask'] = attention_mask
    return result

ds = ds.map(lambda x: encode(x, 'completion'), batched=True, num_proc=30, remove_columns=ds.column_names)
ds = ds.map(group_texts_with_padding, batched=True, num_proc=30, remove_columns=ds.column_names)
zhihu_ds = zhihu_ds.map(encode_instruction, batched=True, num_proc=30, remove_columns=zhihu_ds.column_names)
zhihu_ds = zhihu_ds.map(group_texts_with_padding, batched=True, num_proc=30, remove_columns=zhihu_ds.column_names)
ff_ds = ff_ds.map(encode, batched=True, num_proc=30, remove_columns=ff_ds.column_names)
ff_ds = ff_ds.map(group_texts_with_padding, batched=True, num_proc=30, remove_columns=ff_ds.column_names)
novel_ds = novel_ds.map(encode, batched=True, num_proc=30, remove_columns=novel_ds.column_names)
novel_ds = novel_ds.map(group_texts_with_padding, batched=True, num_proc=30, remove_columns=novel_ds.column_names)
# web_ds = web_ds.map(encode, batched=True, num_proc=30, remove_columns=web_ds.column_names)
# web_ds = web_ds.map(group_texts_with_padding, batched=True, num_proc=30, remove_columns=web_ds.column_names)
ds = concatenate_datasets([ds, ff_ds, novel_ds, zhihu_ds])
print(ds)


def main(**kwargs):
    from transformers import TrainingArguments, Trainer, TrainerCallback, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
    from datetime import datetime

    # TF32 设置（建议启用）
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.set_float32_matmul_precision("high")

    # print(sample(model, '中国首都是哪?'))
    # buddygpt.FLASH = 1
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    class SampleTextCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step % 100 == 0:
                prompt = "中国首都是哪?"
                gen_text = sample(model, prompt, max_length=128)
                print(f"\n[Sample generated at step {state.global_step}]:\n{gen_text}\n")

            if state.global_step % 100 == 0:
                prompt = "which is the capital of china?"
                gen_text = sample(model, prompt, max_length=128)
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
        learning_rate=2e-5,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.2,
        per_device_train_batch_size=5,
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
        # eval_strategy="steps",  # or eval_strategy="steps" in newer versions
        # eval_steps=500,              # Correct parameter name
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        callbacks=[SampleTextCallback],
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
