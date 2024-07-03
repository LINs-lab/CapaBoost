# Increasing Model Capacity for Free: A Simple Strategy for Parameter Efficient Fine-Tuning [ICLR 2024]

<a href="https://www.linkedin.com/in/haobo-song-28a969167/?originalSubdomain=ch">Haobo Song*</a>, <a href="https://marcelluszhao.github.io/">Hao Zhao*</a>, <a href="https://scholar.google.com/citations?user=QSUfCpQAAAAJ&hl=en">Soumajit Majumder</a>, <a href="https://tlin-taolin.github.io/">Tao Lin</a>

**Paper:** [http://arxiv.org/abs/2407.01320](http://arxiv.org/abs/2407.01320) (accepted at ICLR 2024)

> TL;DR: We pursue the effective use of parameter-efficient modules (e.g., LoRA and Adapter) from the perspective of boosting model capacity.
> - We introduce CAPABOOST, a plugin framework that seamlessly integrates with various PEFT methods, such as Adapters and LoRA. Through model pruning and weight-sharing, CAPABOOST
enhances model rank without incurring additional costs.
> - Extensive results unequivocally demonstrate that CAPABOOST outperforms all state-of-the-art baselines while significantly reducing parameter count and maintaining the same or fewer FLOPs.

This dicrectory contains code to run all the RoBERTa experiments in our paper. The code is based on [adapterhub](https://github.com/adapter-hub/adapter-transformers).

## Environments

We use Pytorch 1.11.0+cu113 and Nvidia RTX 4090.
Before running the code, please install the requirements and the propetl package by
```
python install -r requirements
python install .
```

## How to run the models
To reproduce the experiment in the paper Table 1, you can simply run the following 3 shell (Adapter, LoRA and prefix).

Model | CoLA | SST-2 | MRPC | QQP | STS-B | MNLI | QNLI | RTE | Avg
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
ProAdapter | 65.43 | 94.15 | 88.24/91.41 | 89.40/86.04 | 91.34/90.95 | 86.53 | 92.58 | 76.50 | 86.6
ProLoRA | 61.81 | 94.00 | 87.42/91.00 | 88.85/85.22 | 90.48/90.47 | 85.73 | 91.05 | 63.79 | 84.53
ProPrefix | 62.16 | 93.62 | 88.73/91.80 | 87.59/83.71 | 90.92/90.83 | 85.30 | 91.75 | 72.66 | 85.37

```
# propetl adapter
bash scripts/run_adapter.sh
# propetl LoRA
bash scripts/run_lora.sh
# propetl prefix tuning
bash scripts/run_prefix.sh
```

## Citation
If you find this useful in your research, please consider citing:
```
@inproceedings{haobo2023increasing,
  title={Increasing Model Capacity for Free: A Simple Strategy for Parameter Efficient Fine-tuning},
  author={Haobo, SONG and Zhao, Hao and Majumder, Soumajit and Lin, Tao},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
