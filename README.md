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


this **whitening matrix** satisfies:

![alt text](docs/whitening.png)


## model design


## experiment design

Baseline: original VGGT

