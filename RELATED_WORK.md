# Related Work

## ASVD

Activation-aware SVD (basically normalizing the activation):

![alt text](docs/ASVD_1.png)

where a **scaling matrix** $S$ (which is diagonal) can be derived as follows ($X$ is the input activation):

![alt text](docs/ASVD_2.png)

training objective (shared by SVD-LLM too):

![alt text](docs/objective.png)

## SVD-LLM (V2)

adaptive compression ratio + two rounds SVD for finetuning:

![alt text](docs/SVD-LLM_V2.png)

| assigned ratio | min truncation loss |
| ---- | ----- |
| ![alt text](docs/ratio_formula.png) | ![alt text](docs/minimum_loss.png)|

## Householder Transformation

Apply householder transformation to achieve flexible bottleneck dimensionality:

![alt text](docs/householder.png)

## SVDFormer

A shallow feature extraction module with a feature enhancement module (with SVD-attention):

![alt text](docs/SVDFormer.png)

# VGGT inference

![alt text](toy_output/depth_b0_s1.png)

![alt text](toy_output/depth_b0_s0.png)

![alt text](toy_output/depth_b0_s2.png)