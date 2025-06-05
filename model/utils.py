from datasets import load_dataset
from transformers import AutoTokenizer

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