from transformers import AutoTokenizer
from buddygpt import GPTConfig
from buddygpt import BuddyGPT
import buddygpt
import torch
import torch.nn.functional as F
import torch.nn as nn


def load_model_tokenizer(tokenizer_name, seq_len,device):
    def print_parameters(model):
        num_param = sum(
            [param.numel() for param in model.parameters() if param.requires_grad]
        )
        print(f"total param {num_param/1024/1024}m")

    # uer/gpt2-chinese-cluecorpussmall
    # Qwen/Qwen3-0.6B
    # gpt2
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token
    config = GPTConfig(
        n_block=seq_len,
        n_layer=16,
        n_head=16,
        n_kv_head=8,
        n_embed=1536,
        n_vocab=151669,
        pad_token_id=151643,
        eos_token_id=151645,
        tie_word_embeddings=True,
    )
    model = BuddyGPT(config)
    print(tokenizer.pad_token, tokenizer.pad_token_id)
    print(tokenizer.eos_token, tokenizer.eos_token_id)
    print(model)
    print_parameters(model)
    model.to(device)
    return tokenizer, model


def sample(tokenizer, model, query, max_length=50):
    input_ids = tokenizer.encode(query, return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_length,
    )
    gen_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return gen_text



# print(sum([p.numel() for p in model.parameters()]) / 1024 / 1024)


## load dataset
def load_dataset(tokenizer, num_proc, seq_len):
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
            "input_ids": [concatenated[i:i+seq_len] for i in range(0, total_length, seq_len)]
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

    ds = ds.map(lambda x: encode(x, 'completion'), batched=True, num_proc=num_proc, remove_columns=ds.column_names)
    ds = ds.map(group_texts_with_padding, batched=True, num_proc=num_proc, remove_columns=ds.column_names)
    zhihu_ds = zhihu_ds.map(encode_instruction, batched=True, num_proc=num_proc, remove_columns=zhihu_ds.column_names)
    zhihu_ds = zhihu_ds.map(group_texts_with_padding, batched=True, num_proc=num_proc, remove_columns=zhihu_ds.column_names)
    ff_ds = ff_ds.map(encode, batched=True, num_proc=num_proc, remove_columns=ff_ds.column_names)
    ff_ds = ff_ds.map(group_texts_with_padding, batched=True, num_proc=num_proc, remove_columns=ff_ds.column_names)
    novel_ds = novel_ds.map(encode, batched=True, num_proc=num_proc, remove_columns=novel_ds.column_names)
    novel_ds = novel_ds.map(group_texts_with_padding, batched=True, num_proc=num_proc, remove_columns=novel_ds.column_names)
    # web_ds = web_ds.map(encode, batched=True, num_proc=30, remove_columns=web_ds.column_names)
    # web_ds = web_ds.map(group_texts_with_padding, batched=True, num_proc=30, remove_columns=web_ds.column_names)
    ds = concatenate_datasets([ds, ff_ds, novel_ds, zhihu_ds])
    print(ds)
    return ds

def train(ds, tokenizer, model, output_dir, per_device_train_batch_size, gradient_accumulation_steps, flash_attn):
    from transformers import TrainingArguments, Trainer, TrainerCallback, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
    from datetime import datetime

    # TF32 设置（建议启用）
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.set_float32_matmul_precision("high")

    # print(sample(model, '中国首都是哪?'))
    if flash_attn:
        buddygpt.FLASH = 1
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
        per_device_train_batch_size=per_device_train_batch_size,
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
        gradient_accumulation_steps=gradient_accumulation_steps,
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

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='outputs/buddygpt-qwen3')
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--flash_attn", type=bool, default=True)
    parser.add_argument("--ds_num_proc", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer, model = load_model_tokenizer(tokenizer_name='Qwen/Qwen3-0.6B', seq_len=args.block_size, device=device)
    sample(tokenizer, model, '中国首都是哪?')
    ds = load_dataset(tokenizer, num_proc=args.ds_num_proc, seq_len=args.block_size)
    train(
        ds=ds,
        tokenizer=tokenizer,
        model=model,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        flash_attn=args.flash_attn,
    )


if __name__ == "__main__":
    main()
