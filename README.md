## buddygpt

> *train llm from scratch especially for the chinese language*
> with RoPE, GQA, SWiGLU, RMSNorm, weight-tying, FLASH-ATTENTION

|model|Tied Embedding|RoPE|Q-head|KV-head|n_embed|n_layer|seq_len|batch_size(token)|loss|
|-|-|-|-|-|-|-|-|-|-|
|buddygpt-0.1b|✅|✅|16|8|768|8|1024|20*64k|3.5766|
|buddygpt-0.3b|✅|✅|16|8|1024|24|1024|20*64k|-|
|buddygpt-0.4b|✅|✅|16|8|**1024**|**32**|1024|**20*64k**|3.6754|


## implementation

```mermaid
graph LR
    WIKI[wikipedia-1.8b] --> Pretrain[buddygpt-base-0.4b]
    FIRFLY[firefly-13b] --> Pretrain
    Pretrain --> SFT[SFT]
    SFT --> RLHF[RLHF]
    RLHF --> EVAL[评估]
    EVAL --> END[结束]
```

## pretrain
#### dataset
本次训练的预训练预料都来自[Hugging Face](https://huggingface.co/)，主要包含以下几个经典的中文数据集，大约有35B左右Token，详细数据集如下：

| 中文预训练语料    | 链接                                                         | 描述                                            |
| ----------------- | ------------------------------------------------------------ | ----------------------------------------------- |
| Ultra-FineWeb | [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) | Ultra-FineWeb is a large-scale, high-quality, and efficiently-filtered dataset(1T[en]+120B[zh]) |
| Firefly pretrain | [firefly-pretrain](https://huggingface.co/datasets/YeungNLP/firefly-pretrain-dataset) | Firefly 模型训练的部分中文数据(4.7B) |
| Mxode/Chinese-Instruct |[Chinese-Instruct](https://huggingface.co/datasets/Mxode/Chinese-Instruct) | 中文指令微调数据集(100B) |


#### summary

- buddygpt-0.1b-base-zh
![step8600](static/step8600-zh.png)
![step8600](static/step8600-en.png)
![train_metrics](static/01b-train-metrics.png)

- buddygpt-0.2b-base-zh
![buddygpt-0.2b-base-zh](static/buddygpt-0.2b-base-zh.png)

- buddygpt-0.4b-base-zh
![buddygpt-0.4b-base-zh](static/buddygpt-0.2b-base-zh.png)
![step1200](static/step1200.png)
![step2800](static/step2800.png)
![step3800](static/step3800.png)
![step7600](static/step7600.png)
![step10000](static/step10000.png) 可以看到开始有北京了，这时候loss=3.8
 
#### metrics
|model|cmmlu|mmlu|ceval|gpqa|ifeval|aime24|math-500|livecodebench|
|-|-|-|-|-|-|-|-|-|
|[buddygpt-0.1b-base](https://huggingface.co/learn2pro/buddygpt-0.1b-base)|*25.38*|*24.64*|*24.29*|25.38|25.06|0.1|0.1|0.1|
|deepseek-r1|-|**90.8**|-|59.1|**86.1**|39.2|**90.2**|37.6|
|deepseek-v3|**88.8**|88.5|**90.1**|59.1|**86.1**|39.2|**90.2**|37.6|
|qwen2.5-0.5b|41.44|45.2|39.23|-|-|-|32.44|-|
|qwen3-0.6b|35.29|37.56|37.6|-|-|-|32.44|-|
|buddy-0.3b-base|25.6|25.04|28.6|-|-|-|32.44|-|
|buddy-0.3b-chat|25.37|25.04|25.85|-|-|-|32.44|-|


## SFT

SFT指令微调预料都来自[Hugging Face](https://huggingface.co/)，主要包含以下几个经典的SFT数据集，大约有400w条，详细数据集如下：

| SFT微调数据 | 链接                                                         | 描述                                       |
| ----------- | ------------------------------------------------------------ | ------------------------------------------ |
| Mxode/Chinese-Instruct-Lite |[Chinese-Instruct-Lite](https://huggingface.co/datasets/Mxode/Chinese-Instruct-Lite/viewer/general) | 一个全新的简化数据集 |
| Belle       | [Belle_train](https://huggingface.co/datasets/BelleGroup/train_2M_CN) | 包含约200万条由BELLE项目生成的中文指令数据 |
| YeungNLP/moss-003-sft-data |[moss-003-sft-data](https://huggingface.co/datasets/YeungNLP/moss-003-sft-data)|YeungNLP|
| shareAI/ShareGPT-Chinese-English-90k |[ShareGPT-Chinese-English-90k](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k) | A high-quality Chinese-English parallel bilingual human-machine QA dataset |
| FuseChat-3.0-SFT-Data       | [FuseChat-3.0-SFT-Data](https://huggingface.co/datasets/FuseAI/FuseChat-3.0-SFT-Data) | FuseChat-3.0: Preference Optimization for Implicit Model Fusion |

#### metrics
|model|cmmlu|mmlu|ceval|gpqa|ifeval|aime24|math-500|livecodebench|
|-|-|-|-|-|-|-|-|-|
|[buddygpt-0.1b-base](https://huggingface.co/learn2pro/buddygpt-0.1b-base)|*25.38*|*24.64*|*24.29*|25.38|25.06|0.1|0.1|0.1|
|[buddygpt-0.1b-chat](https://huggingface.co/learn2pro/buddygpt-0.1b-chat)|*25.19*|*24.47*|*31.75*|25.38|25.06|0.1|0.1|0.1|
|deepseek-r1|-|**90.8**|-|59.1|**86.1**|39.2|**90.2**|37.6|
|deepseek-v3|**88.8**|88.5|**90.1**|59.1|**86.1**|39.2|**90.2**|37.6|
|qwen2.5-0.5b|41.44|45.2|39.23|-|-|-|32.44|-|
|qwen3-0.6b|35.29|37.56|37.6|-|-|-|32.44|-|


## RLHF

来源于开源DPO数据集，详细数据集如下：

| DPO微调数据 | 链接                                                         | 描述                                       |
| ----------- | ------------------------------------------------------------ | ------------------------------------------ |
| FuseAI/FuseChat-3.0-DPO-Data       | [FuseAI/FuseChat-3.0-DPO-Data](https://huggingface.co/datasets/FuseAI/FuseChat-3.0-DPO-Data) | 包含约200万条由BELLE项目生成的中文指令数据 |
| Hello-SimpleAI/HC3-Chinese     | [Hello-SimpleAI/HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese) | 流萤开源模型SFT数据集                      |
|YeungNLP/ultrafeedback_binarized|[YeungNLP/ultrafeedback_binarized](https://huggingface.co/datasets/YeungNLP/ultrafeedback_binarized)|YeungNLP DPO|
|HuggingFaceH4/ultrafeedback_binarized|[HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)| HuggingFaceH4/ultrafeedback_binarized |


## code structure

- model: the model structure code
- pretrain: pretrain workflow
- sft: finetune workflow
- rlhf: rlhf with DPO https://arxiv.org/pdf/2305.18290
- eval: evaluate tool with [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)

## script

- pretrain: 
```
cd pretrain && accelerate launch --config_file ptrain.yaml --num_processes=1 pretrain.py
```
- eval: 
```shell
export PYTHONPATH=$(pwd):$PYTHONPATH
lm_eval --model hf \
    --model_args pretrained=learn2pro/buddygpt-0.4b-base-zh,dtype="bfloat16" \
    --tasks cmmlu,gpqa \
    --device cuda:0 \
    --batch_size 8 \
    --num_fewshot 2 \
    --output_path results/cmmlu_2shot_log \
    --log_samples

lm_eval --model hf \
    --model_args pretrained=learn2pro/buddygpt-0.4b-base-zh,dtype="bfloat16" \
    --tasks cmmlu \
    --device cuda:0 \
    --batch_size 8 \
    --num_fewshot 2 \
    --output_path results/cmmlu_2shot_log \
    --log_samples


lm_eval --model hf \
    --model_args pretrained=outputs/buddysft-qwen3,dtype="bfloat16" \
    --tasks cmmlu \
    --device cuda:0 \
    --batch_size 8

lm_eval --model hf \
    --model_args pretrained=qwen/qwen3-0.6b,dtype="bfloat16" \
    --tasks cmmlu \
    --device cuda:0 \
    --batch_size 32

all_proxy= python eval/eval.py

```

- serve by transformer
```
transformers chat learn2pro/buddygpt-0.1b-chat
```

- push_to_hub:
```
huggingface-cli login
huggingface-cli repo create buddygpt-0.3b-chat --type model
huggingface-cli upload learn2pro/buddygpt-0.3b-chat .
```

- push to modelscope:
```
modelscope login
all_proxy= modelscope modelcard -act create -mid learn2pro/buddygpt-0.2b-chat-zh -ch learn2pro/buddygpt-0.2b-chat-zh
all_proxy= modelscope upload learn2pro/buddygpt-0.2b-chat-zh .
```


