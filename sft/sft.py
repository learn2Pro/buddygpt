from transformers import AutoTokenizer, AutoModelForCausalLM
import model.modeling_buddygpt
import model.modeling_tinyllm
import torch
import torch.nn.functional as F
import torch.nn as nn


system_prompt = 'You are a helpful assistant, created by learn2pro!'
def print_parameters(model):
    num_param = sum(
        [param.numel() for param in model.parameters() if param.requires_grad]
    )
    print(f"total param {num_param/1024/1024}m")



def load_tokenizer_model(model_id, seq_len, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    print(len(tokenizer))
    print(model)
    print_parameters(model)
    return tokenizer, model

def do_sample(tokenizer, model, prompt, max_new_tokens=128):
    messages = [
        {"role":"system", "content": system_prompt},
        {"role":"user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(    
        messages,
        tokenize=False,              # return plain text
        add_generation_prompt=True  # adds trailing "Assistant:" or equivalent)
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
    )
    gen_text = tokenizer.decode(output[0], skip_special_tokens=False)
    return gen_text

def load_dataset(tokenizer, num_proc:int, seq_len:int):
    from datasets import load_dataset, concatenate_datasets
    # dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    # dataset = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT", split="train")
    # https://github.com/yangjianxin1/Firefly?tab=readme-ov-file
    # shareAI/ShareGPT-Chinese-English-90k
    # YeungNLP/moss-003-sft-data
    stem_ds = load_dataset("zake7749/chinese-sft-stem-zh-hant", split="train")
    sharegpt_ds = load_dataset("YeungNLP/moss-003-sft-data", split="train")
    tiger_rs_ds = load_dataset("TigerResearch/sft_zh", split="train")
    
    def format_chat_template(example):
        output_texts = []
        for i in range(len(example["conversations"])):
            messages = []
            conversation = example["conversations"][i]
            # con_json = json.loads(conversation)
            # user_content = f'instruction:{example["instruction"][i]}'+("" if example["input"][i] else f'\ninput:{example["input"][i]}')
            messages.append({"role":"system", "content": system_prompt})
            messages.append({"role":"user", "content": conversation[0]["value"]})
            messages.append({"role":"assistant", "content": f'{conversation[1]["value"]}'})
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            output_texts.append(text)
        result = {}
        result['text'] = output_texts
        return result
    
    def format_chat_template2(example):
        output_texts = []
        for i in range(len(example["conversation"])):
            messages = []
            conversation = example["conversation"][i]
            messages.append({"role":"system", "content": system_prompt})
            for item in conversation:
                messages.append({"role":"user", "content": item["human"]})
                messages.append({"role":"assistant", "content": f'{item["assistant"]}'})
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            output_texts.append(text)
        result = {}
        result['text'] = output_texts
        return result

    def format_chat_template3(example):
        output_texts = []
        for i in range(len(example["instruction"])):
            messages = []
            instruction = example["instruction"][i]
            input_prompt = example["input"][i]
            output = example["output"][i]
            messages.append({"role":"system", "content": system_prompt})
            messages.append({"role":"user", "content": f'{instruction}\n{input_prompt}'})
            messages.append({"role":"assistant", "content": f'{output}'})
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            output_texts.append(text)
        result = {}
        result['text'] = output_texts
        return result

    def mask_instruction(example, split_word = '<|im_start|>assistant'):
        input_ids = []
        labels = []
        attention_mask = []
        for i in range(len(example["text"])):
            inst_str, resp_str = example["text"][i].split(split_word, maxsplit=1)
            # if i == 0:
            #     print(inst_str, '\n<split>\n', resp_str)
            instruction = tokenizer.encode(inst_str + split_word, add_special_tokens=True, truncation=True, max_length=seq_len)
            response = tokenizer.encode(resp_str, add_special_tokens=False, truncation=True, max_length=seq_len)
            input_ids = instruction + response
            labels = [tokenizer.pad_token_id] * len(instruction) + response
            
            pad_len = seq_len - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [tokenizer.pad_token_id] * pad_len
            labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
    
            input_ids = torch.LongTensor(input_ids)
            labels = torch.LongTensor(labels)
            attention_mask = input_ids.ne(tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        
        
    stem_ds = stem_ds.map(format_chat_template, batched=True, num_proc=30, remove_columns=stem_ds.column_names)
    # stem_ds = stem_ds.map(mask_instruction, batched=True, num_proc=30, remove_columns=stem_ds.column_names)
    
    sharegpt_ds = sharegpt_ds.map(format_chat_template2, batched=True, num_proc=30, remove_columns=sharegpt_ds.column_names)
    # sharegpt_ds = sharegpt_ds.map(mask_instruction, batched=True, num_proc=30, remove_columns=sharegpt_ds.column_names)
    
    tiger_rs_ds = tiger_rs_ds.map(format_chat_template3, batched=True, num_proc=30, remove_columns=tiger_rs_ds.column_names)
    # tiger_rs_ds = tiger_rs_ds.map(mask_instruction, batched=True, num_proc=30, remove_columns=tiger_rs_ds.column_names)
    
    ds = concatenate_datasets([tiger_rs_ds, stem_ds, sharegpt_ds])
    print(ds)
    return ds

def train(ds, tokenizer, model, output_dir, per_device_train_batch_size, gradient_accumulation_steps, seq_len, num_proc):
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
    
    class SampleTextCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step % 100 == 0:
                prompt = "中国首都是哪?"
                gen_text = do_sample(tokenizer, model, prompt)
                print(f"\n[Sample generated at step {state.global_step}]:\n{gen_text}\n")
            if state.global_step % 100 == 0:
                prompt = "which is the capital of China?"
                gen_text = do_sample(tokenizer, model, prompt)
                print(f"\n[Sample generated at step {state.global_step}]:\n{gen_text}\n")
    
    # 定义 DataCollator，仅对 assistant 区域计算 loss
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template="<|im_start|>user",  # 开始 loss 的位置
        response_template="<|im_start|>assistant",  # 如果你想从 assistant 开始一直算 loss，可以省略
        mlm=False,
    )

    
    args = SFTConfig(
        output_dir=output_dir,
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=1,
        logging_steps=50,
        save_steps=100,
        save_total_limit=10,
        bf16=True,
        max_grad_norm=1.0,
        dataset_text_field='text',
        dataset_num_proc=num_proc,
        max_seq_length=seq_len,
    )
    
    trainer = SFTTrainer(
        model,
        train_dataset=ds,
        args=args,
        # formatting_func=lambda x: formatting_prompts_func(tokenizer, x),
        data_collator=collator,
        callbacks=[SampleTextCallback],
    )
    
    trainer.train()
    trainer.save_model(output_dir)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default='learn2pro/buddygpt-0.2b-base-zh')
    parser.add_argument("--output_dir", type=str, default='outputs/buddygpt-0.2b-chat-zh')
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
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
    tokenizer, model = load_tokenizer_model(model_id=args.model_id, seq_len=args.block_size, device=device)
    do_sample(tokenizer, model, '中国首都是哪?')
    ds = load_dataset(tokenizer, num_proc=args.ds_num_proc, seq_len=args.block_size)
    train(
        ds=ds,
        tokenizer=tokenizer,
        model=model,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seq_len=args.block_size,
        num_proc=args.ds_num_proc,
    )


if __name__ == "__main__":
    main()