from transformers import AutoTokenizer
from model.configuration_buddygpt import BuddyGPTConfig
from model.modeling_buddygpt import BuddyGPTForCausalLM
import torch
import torch.nn.functional as F
import torch.nn as nn

from accelerate import Accelerator
accelerator = Accelerator()

def print_parameters(model):
    num_param = sum(
        [param.numel() for param in model.parameters() if param.requires_grad]
    )
    print(f"total param {num_param/1024/1024}m")


def load_tokenizer_model(tokenizer_name, seq_len, n_layer, n_embed, n_head, attn_impl='sdpa', device='cuda'):

    # uer/gpt2-chinese-cluecorpussmall
    # Qwen/Qwen3-0.6B
    # gpt2
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token
    config = BuddyGPTConfig(
        vocab_size=len(tokenizer),
        hidden_size=n_embed,
        intermediate_size=n_embed * 2,
        num_hidden_layers=n_layer,
        num_attention_heads=n_head,
        num_key_value_heads=n_head // 2,
        num_seq_len=seq_len,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,
        _attn_implementation=attn_impl,
    )
    model = BuddyGPTForCausalLM(config)
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



## load dataset
def load_dataset(tokenizer, num_proc, seq_len):
    from datasets import load_dataset, concatenate_datasets, load_from_disk
    import os

    data_cache_dir = 'data/pretrain_processed'

        
    def group_texts(examples):
        from itertools import chain
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= seq_len:
            total_length = (total_length // seq_len) * seq_len
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated_examples.items()
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
    
    if accelerator.is_main_process:
        if not os.path.exists(data_cache_dir):
            # 50m model need 20*50m = 1B token
            # 100m model need 20*100m = 2B token
            # 200m model need 20*200m = 4B token
            # 500m model need 20*500m = 10B token
            # Total tokens: 1872137976
            # 1.8B token
            # ds = load_dataset("pleisto/wikipedia-cn-20230720-filtered", split="train")
            # zhihu instruction 0.47b
            # zhihu_ds = load_dataset("wangrui6/Zhihu-KOL", split="train")
            # 10B token
            # web_ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train")
            web_ds = load_dataset("openbmb/Ultra-FineWeb", split="zh").filter(lambda x: x['score'] >= 0.7)
            # wikipedia ds
            wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train")
            # firefly ds
            # 4.7B token
            # ff_ds = load_dataset("YeungNLP/firefly-pretrain-dataset", split="train")
            # novel ds
            # 8.4b
            # novel_ds = load_dataset("wdndev/webnovel-chinese", split="train")

            # ds = ds.map(lambda x: encode(x, 'completion'), batched=True, num_proc=num_proc, remove_columns=ds.column_names)
            # ds = ds.map(group_texts, batched=True, num_proc=num_proc, remove_columns=ds.column_names)
            # zhihu_ds = zhihu_ds.map(encode_instruction, batched=True, num_proc=num_proc, remove_columns=zhihu_ds.column_names)
            # zhihu_ds = zhihu_ds.map(group_texts, batched=True, num_proc=num_proc, remove_columns=zhihu_ds.column_names)
            # ff_ds = ff_ds.map(encode, batched=True, num_proc=num_proc, remove_columns=ff_ds.column_names)
            # ff_ds = ff_ds.map(group_texts, batched=True, num_proc=num_proc, remove_columns=ff_ds.column_names)
            # novel_ds = novel_ds.map(encode, batched=True, num_proc=num_proc, remove_columns=novel_ds.column_names)
            # novel_ds = novel_ds.map(group_texts, batched=True, num_proc=num_proc, remove_columns=novel_ds.column_names)
            web_ds = web_ds.map(lambda x: encode(x, 'content'), batched=True, num_proc=num_proc, remove_columns=web_ds.column_names)
            web_ds = web_ds.map(group_texts, batched=True, num_proc=num_proc, remove_columns=web_ds.column_names)
            wiki_ds = wiki_ds.map(lambda x: encode(x, 'text'), batched=True, num_proc=num_proc, remove_columns=wiki_ds.column_names)
            wiki_ds = wiki_ds.map(group_texts, batched=True, num_proc=num_proc, remove_columns=wiki_ds.column_names)
            # ds = concatenate_datasets([web_ds, ds, ff_ds, novel_ds, zhihu_ds])
            ds = concatenate_datasets([web_ds, ds])
            print(ds)
            ds.save_to_disk(data_cache_dir)
        else:
            print("Using cached data")
    
    accelerator.wait_for_everyone()
    ds = load_from_disk(data_cache_dir)


    return ds

def train(ds, tokenizer, model, output_dir, per_device_train_batch_size, gradient_accumulation_steps, flash_attn, sample_step):
    from transformers import TrainingArguments, Trainer, TrainerCallback, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
    from datetime import datetime

    # TF32 设置（建议启用）
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.set_float32_matmul_precision("high")

    # print(sample(model, '中国首都是哪?'))
    # if flash_attn:
    #     buddygpt.FLASH = 1
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    class SampleTextCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step % sample_step == 0:
                prompt = "中国首都是哪?"
                gen_text = sample(tokenizer, model, prompt, max_length=128)
                print(f"\n[Sample generated at step {state.global_step}]:\n{gen_text}\n")

            if state.global_step % sample_step == 0:
                prompt = "which is the capital of china?"
                gen_text = sample(tokenizer, model, prompt, max_length=128)
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
        learning_rate=2e-4,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.2,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        logging_steps=sample_step // 2,
        save_steps=sample_step,
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
    parser.add_argument("--output_dir", type=str, default='outputs/buddygpt-0.4b')
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--n_layer", type=int, default=24)
    parser.add_argument("--n_embed", type=int, default=1536)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--attn_impl", type=str, default='sdpa')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--flash_attn", type=bool, default=True)
    parser.add_argument("--sample_step", type=int, default=100)
    parser.add_argument("--ds_num_proc", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer, model = load_tokenizer_model(
        tokenizer_name='Qwen/Qwen3-0.6B', 
        seq_len=args.block_size, 
        n_layer=args.n_layer, 
        n_embed=args.n_embed, 
        n_head=args.n_head, 
        attn_impl=args.attn_impl,
        device=device,
    )
    sample(tokenizer, model, '中国首都是哪?')
    ds = load_dataset(tokenizer, num_proc=args.ds_num_proc, seq_len=args.block_size)
    train(
        ds=ds,
        tokenizer=tokenizer,
        model=model,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        flash_attn=args.flash_attn,
        sample_step=args.sample_step,
    )


if __name__ == "__main__":
    main()
