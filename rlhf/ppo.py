from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import model.modeling_buddygpt
import model.modeling_tinyllm
import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_dataset
from trl.core import LengthSampler

system_prompt = 'You are a helpful assistant, created by learn2pro!'
def print_parameters(model):
    num_param = sum(
        [param.numel() for param in model.parameters() if param.requires_grad]
    )
    print(f"total param {num_param/1024/1024}m")


def load_tokenizer_model(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id).to(device)
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

def load_ppo_dataset(tokenizer, num_proc:int, seq_len:int):
    ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    
    def tokenize(sample):
        prompt = sample["prompt"]
        sample["input_ids"] = tokenizer.encode(prompt, truncation=True, max_length=seq_len)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, num_proc=num_proc)
    ds.set_format(type="torch")
    print(ds)
    return ds

def train(ds, tokenizer, model, output_dir, per_device_train_batch_size, gradient_accumulation_steps, seq_len, num_proc):
    
    ppo_config = PPOConfig(
        model_name=output_dir,
        learning_rate=1e-5,
        batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimize_cuda_cache=True,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=ds,
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": 100_000,
    }
    output_min_length = 32
    output_max_length = seq_len
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    for epoch, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        #### Get response from SFTModel
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        #### Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        
        # This is a placeholder for a reward model.
        # In a real scenario, you would have a reward model that scores the generated responses.
        # For this example, we'll use a simple length-based reward.
        rewards = [torch.tensor(len(r)) for r in batch["response"]]

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        
        if epoch > 100:
            break

    ppo_trainer.save_pretrained(output_dir)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default='learn2pro/buddygpt-0.2b-chat-zh')
    parser.add_argument("--output_dir", type=str, default='outputs/buddygpt-0.2b-ppo-zh')
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
    ds = load_ppo_dataset(tokenizer, num_proc=args.ds_num_proc, seq_len=args.block_size)
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
