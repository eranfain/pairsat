# PAIRSAT - Integrating Preference-Based Signals for User Satisfaction Estimation in Dialogue Systems

This is the Pytorch implementation of our work: **PAIRSAT: Integrating Preference-Based Signals for User Satisfaction Estimation in Dialogue Systems**.

## Abstract
Predicting user satisfaction in dialogue systems is critical for improving conversational AI quality and user experience. Although supervised labels indicating satisfaction levels are often scarce and costly to collect, implicit preference signals such as accepted versus rejected conversation variants are more abundant but lack direct numerical scores. In this work, we propose a unified approach for user satisfaction estimation that integrates both explicit satisfaction labels and implicit preference data. We reformulate satisfaction prediction as a bounded regression task on a continuous scale, allowing fine-grained modeling of satisfaction levels in [-1, 1]. To exploit preference data, we integrate a pairwise ranking loss that encourages higher predicted satisfaction for accepted conversations over rejected ones. Our model named $PAIRSAT$ jointly optimizes regression on labeled data and ranking on preference pairs using a transformer-based encoder. Experiments demonstrate that our model outperforms purely supervised baselines, highlighting the value of leveraging implicit feedback for satisfaction estimation in dialogue systems.

## Model
We use a Transformer-based encoder to model the conversational context, followed by a multi-layer perceptron (MLP) head with a $\tanh$ activation to predict a bounded satisfaction score within $[-1, 1]$. During training, batches are sampled from both labeled and preference datasets, and gradients are computed jointly.

## How to use?
**Recommended**: we used a g5.xlarge instance on AWS with a single A10G GPU

1. Clone the project
2. Create a venv with python==3.10 and install requirements:
```shell
conda create -n pairsat python=3.10
conda activate pairsat
pip install -r requirements.txt
```
3. Clone the anthropic_hh dataset to the `dataset/anthropic_hh` folder (follow the `readme.txt` file within the folder)
4. To run PRAISE algorithm (baseline method), you should have an Azure environment with the following models enabled: `gpt-4-1106-preview`, `gpt-3.5-turbo-0125` and `text-embedding-3-large`. You should setup the following env variables:
```shell
export AZURE_OPENAI_API_KEY=<YOUR_AZURE_OPENAI_API_KEY>
export COMPLETION_ENDPOINT_URL=<YOUR_COMPLETION_ENDPOINT_URL>
export EMBEDDING_ENDPOINT_URL=<YOUR_EMBEDDING_ENDPOINT_URL>
export COMPLETION_API_VERSION=<YOUR_COMPLETION_API_VERSION>
export EMBEDDING_API_VERSION=<YOUR_EMBEDDING_API_VERSION>
```
5. Running PRAISE algorithm:
```shell
python praise_exp.py --dataset_name mwoz
```
6. Running PAIRSAT (our method):
```shell
python pairsat_exp.py --dataset_name mwoz
```
7. To run our zero-shot and few-shot baselines, use the `gpt-zero-shot-few-shot.ipynb` notebook.

## Reference
Our metrics and data loading implementations are based on [ASAP](https://github.com/smartyfh/asap#). We also leveraged the datasets preprocessed by ASAP.

## Citation
If you use this repository in your research, please cite our paper:
```bibtext
@inproceedings{fainman2025pairsat,
  title={PAIRSAT: Integrating Preference-Based Signals for User Satisfaction Estimation in Dialogue Systems},
  author={Fainman, Eran and Solomon, Adir and Mokryn, Osnat},
  booktitle={Proceedings of the Nineteenth ACM Conference on Recommender Systems},
  pages={1251--1255},
  year={2025}
}
```