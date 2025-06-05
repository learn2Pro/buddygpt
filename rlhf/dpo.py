from transformers import AutoTokenizer, AutoModelForCausalLM
import model.modeling_buddygpt
import model.modeling_tinyllm
import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_dataset

system_prompt = 'You are a helpful assistant, created by learn2pro!'
def print_parameters(model):
    num_param = sum(
        [param.numel() for param in model.parameters() if param.requires_grad]
    )
    print(f"total param {num_param/1024/1024}m")


def load_tokenizer_model(model_id, device):
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

def load_dpo_dataset(tokenizer, num_proc:int, seq_len:int):
    ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    
    print(ds)
    return ds

def train(ds, tokenizer, model, output_dir, per_device_train_batch_size, gradient_accumulation_steps, seq_len, num_proc):
    from transformers import TrainerCallback
    from trl import DPOTrainer, DPOConfig

    training_args = DPOConfig(
        output_dir, 
        per_device_train_batch_size=per_device_train_batch_size, 
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=1,
        logging_steps=50,
        save_steps=100,
        save_total_limit=10,
        bf16=True,
        max_grad_norm=1.0,
    )


    trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=ds)
    trainer.train()
    trainer.save_model(output_dir)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default='learn2pro/buddygpt-0.2b-chat-zh')
    parser.add_argument("--output_dir", type=str, default='outputs/buddygpt-0.2b-dpo-zh')
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
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
    tokenizer, model = load_tokenizer_model(model_id=args.model_id, device=device)
    do_sample(tokenizer, model, '中国首都是哪?')
    ds = load_dpo_dataset(tokenizer, num_proc=args.ds_num_proc, seq_len=args.block_size)
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