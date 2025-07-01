from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def calc_token(dataset, tokenizer, field='text', field1=None, num_proc=30, batch_size=1024):
    # 设置 padding token（GPT-2 默认没有 pad token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 批量处理函数
    def count_tokens_batch(batch):
        if field1 is None:
            tokens = tokenizer(batch[field], add_special_tokens=False)
        else:
            tokens = tokenizer(batch[field], batch[field1], add_special_tokens=False)
    
        return {"num_tokens": [len(x) for x in tokens["input_ids"]]}

    # 映射到整个数据集，批量处理更高效
    # map 时 batched=True, 实际处理的是一个 batch（列表）
    token_count_dataset = dataset.map(
        count_tokens_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Tokenizing",
    )
    
    # 求和
    total_tokens = sum(token_count_dataset["num_tokens"])
    print(f"总 token 数量: {total_tokens / 1e9:.3f}B")
    return total_tokens
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default='wangrui6/Zhihu-KOL')
    parser.add_argument("--tokenizer", type=str, default='qwen/qwen3-0.6b')
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--field", type=str, default='text')
    parser.add_argument("--field1", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=30)
    parser.add_argument("--ds_batch_size", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # ds = load_dataset(
    #     "parquet",
    #     data_files="data/Ultra-FineWeb/data/ultrafineweb_zh/*.parquet",
    #     split="train[:]",
    #     num_proc=args.num_proc,
    # ).filter(lambda x: float(x['score']) >= 0.85, num_proc=args.num_proc)

    ds = load_dataset(
        "json",
        data_files="data/Chinese-Instruct/**/*.jsonl",
        split="train",
        num_proc=args.num_proc,
    )
    
    # ds = load_dataset(
    #     "parquet",
    #     data_files="data/Ultra-FineWeb/data/ultrafineweb_en/ultrafineweb-en-part-0[2-3][0-9][0-9]-of-2048.parquet",
    #     split="train[:]",
    #     num_proc=args.num_proc,
    # ).filter(lambda x: float(x['score']) >= 0.85, num_proc=args.num_proc)
    
    # ds = load_dataset("zake7749/chinese-sft-stem-zh-hant", split="train")
    calc_token(ds, tokenizer, field=args.field, field1=args.field1, num_proc=args.num_proc, batch_size=args.ds_batch_size)
    # subset = ds.limit(100000)
    # avg_len = calc_token(ds, tokenizer, field=args.field, num_proc=args.num_proc)
    # approx_total = avg_len
    # print(f"估算全数据 token 数量：{approx_total / 1e9:.3f}B")
    

if __name__ == "__main__":
    main()