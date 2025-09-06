# SVD-VGGT: Efficient Visual Geometry Grounded Transformer via Singular Value Decomposition

## ASVD

Activation-aware SVD (basically normalizing the activation):

![alt text](docs/ASVD_1.png)

where a **scaling matrix** $S$ (which is diagonal) can be derived as follows ($X$ is the input activation):

![alt text](docs/ASVD_2.png)

training objective (shared by SVD-LLM too):

![alt text](docs/objective.png)

## SVD-LLM

Motivation: in ASVD, truncating the smallest singular values **does not guarantee** minimal loss, hence we want to achieve a **direct mapping** between singular values and compression loss.

![alt text](docs/comparison.png)

Instead of a simple/naive **scaling matrix** illustrated in [ASVD](docs/ASVD_2.png), we can use a **whitening matrix** $S$ 

![alt text](docs/SVD-LLM.png)

this **whitening matrix** is computed such that it satisfies the following property:

![alt text](docs/whitening.png)

## SVD-LLM (V2)

adaptive compression ratio + two rounds SVD for finetuning:

![alt text](docs/SVD-LLM_V2.png)

| assigned ratio | min truncation loss |
| ---- | ----- |
| ![alt text](docs/ratio_formula.png) | ![alt text](docs/minimum_loss.png)|

## Householder Transformation

coming soon!


## SVDFormer

coming soon!

## Evaluation

| method | GFLOPs (1 forward pass) | camera | depth | point | image matching | downstream (TBD) |
| ----- | ------------------------ | ------ | ----- | ----- | -------------- | ---------------- |
| VGGT | ? | ? | ? | ? | ? | ? |
| VGGT with vanilla-SVD | ? | ? | ? | ? | ? | ? |
| VGGT with ASVD | ? | ? | ? | ? | ? | ? |
| **VGGT with SVD-LLM** | ? | ? | ? | ? | ? | ? |

## Alternative idea

1. SVD-ViT (VGGT is already for multi-task, but maybe ViT has **broader impacts**?)
    1. use the whitening + param update introduced from SVD-LLM
    2. compare with existing ViT-related work like [Efficient Adaptation of Pre-trained Vision Transformer via Householder Transformation](https://arxiv.org/pdf/2410.22952)


## VGGT inference

![alt text](toy_output/depth_b0_s1.png)

![alt text](toy_output/depth_b0_s0.png)

![alt text](toy_output/depth_b0_s2.png)

## SVD-LLM preliminaries

### Truncation-Aware Data Whitening

using the calibration dataset for data whitening:

```bash
CUDA_VISIBLE_DEVICES=<whichever_is_free> python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path . --run_low_resource
```

perplexity evaluation:

```bash
CUDA_VISIBLE_DEVICES=<whichever_is_free> taskset -c 30-40 python SVDLLM.py --step 4 --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt
```

```java
PPL after pruning: {'wikitext2': 7.886700954800093}
Weight Memory: 22004.896484375 MiB
```

efficiency evaluation:

```bash
CUDA_VISIBLE_DEVICES=<whichever_is_free> taskset -c 30-40 python SVDLLM.py --step 5 --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt
```

```java
Total Memory: 28.538090705871582 GB
Weight Memory: 20.503570556640625 GB
Activation Memory: 8.026554107666016 GB
Throughput: 69.48256829185354 tokens/sec
```

### Finetuning with LoRA

update W'u:

```bash
CUDA_VISIBLE_DEVICES=<whichever_is_free> nohup taskset -c 30-40 python utils/LoRA.py --prune_model jeffwan_llama_7b_hf_whitening_only_0.8.pt --data_path yahma/alpaca-cleaned --output_dir ./first_half --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 4 --micro_batch_size 1 --cutoff_len 1024 --group_by_length &
```

```JSON
{'train_runtime': 128586.2054, 'train_samples_per_second': 1.161, 'train_steps_per_second': 0.29, 'train_loss': 1.0868874290876194, 'epoch': 3.0}
```

Immediate evaluation:

```bash
CUDA_VISIBLE_DEVICES=<whichever_is_free> taskset -c 30-40 python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora ./first_half --step 4
```

```java
coming soon!
```

Update W'v:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 taskset -c 30-40 python utils/LoRA.py --prune_model ./first_half/merge.pt --data_path yahma/alpaca-cleaned --output_dir ./second_half --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64
```

Immediate evaluation:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 taskset -c 30-40 python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora ./first_half /first_half --step 4
```

```java
coming soon!
```

Final evaluation:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 taskset -c 30-40 python SVDLLM.py --model_path ./first_half/merge.pt --lora ./second_half --step 4
```
