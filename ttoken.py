from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def calc_token(dataset, tokenizer, field='text', num_proc=30):
    # 设置 padding token（GPT-2 默认没有 pad token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 假设我们要处理字段 "text"
    def count_tokens(example):
        return {"num_tokens": len(tokenizer.encode(example[field]))}
    
    # 映射到整个数据集，批量处理更高效
    token_count_dataset = dataset.map(count_tokens, num_proc=num_proc, batched=False)
    
    # 求和
    total_tokens = sum(token_count_dataset["num_tokens"])
    
    print(f"总 token 数量: {total_tokens/1024/1024/1024}b")
    return total_tokens
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default='wangrui6/Zhihu-KOL')
    parser.add_argument("--tokenizer", type=str, default='qwen/qwen3-0.6b')
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--field", type=str, default='text')
    parser.add_argument("--num_proc", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    ds = load_dataset(args.ds, split=args.split).filter(lambda x: x['score'] >= 0.99)
    calc_token(ds, tokenizer, field=args.field, num_proc=args.num_proc)
    

if __name__ == "__main__":
    main()