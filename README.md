<h1 align="center">(AAAI Alignment Track 2025 Poster) Stream Aligner: Efficient Sentence-Level Alignment via Distribution Induction </h1>

This repository contains the source code for our AAAI Alignment Track 2025 Poster paper "Stream Aligner: Efficient Sentence-Level Alignment via Distribution Induction".

Work done by [PKU-Alignment Team](https://github.com/PKU-Alignment)

## Abstract

The rapid advancement of large language models (LLMs) has led to significant improvements in their capabilities, but also to increased concerns about their alignment with human values and intentions. Current alignment strategies, including adaptive training and inference-time methods, have demonstrated potential in this area. However, these approaches still struggle to balance deployment complexity and capability across various tasks and difficulties. In this work, we introduce the Streaming Distribution Induce Aligner (*Stream Aligner*), a novel alignment paradigm that combines efficiency with enhanced performance in various tasks throughout the generation process. *Stream Aligner* achieves dynamic sentence-level correction by using a small model to learn the preferences of the suffix sentence, iteratively correcting the suffix sentence output by the upstream model, and then using the corrected sentence to replace the suffix sentence in subsequent generations. Compared to Aligner, our experiments demonstrate that *Stream Aligner* reduces reliance on the capabilities of additional models, enhances the reasoning abilities of LLMs, and decreases latency during user interaction. Specifically, *Stream Aligner*-2B model has achieved an improvement of 76.1% in helpfulness, 36.0% in harmlessness on the tested Llama2-70B-chat model, and *Stream Aligner*-8B has achieved an improvement of 3.5% on the math ability of the tested Llama3-70B-chat model.

## Installation

Clone the source code from GitHub:

```bash
git clone https://github.com/htlou/stream-aligner.git
cd stream-aligner
```

Set up the environment:

```bash
conda create -n stream-aligner python=3.10
conda activate stream-aligner
cd train
pip install -e .
```

## Datasets

We open-source the dataset used in our paper. Please refer to our [huggingface repo](https://huggingface.co/datasets/htlou/stream-aligner) for more details.

## Training

`stream-aligner` supports a complete pipeline for Stream Aligner <em>residual correction</em> training.

0. Follow the instructions in section [Installation](#installation) to setup the training environment properly.

1. Download the correct dataset and model, and set the correct path in the `train.sh` script.

2. Run the training script:

```bash
cd train
bash train.sh
```

## Evaluation

Please refer to the `generation` directory for the code used to generate the results for evaluation, and the `evaluation` directory for the code used to evaluate the results.

## Acknowledgment

This repository benefits from [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai), [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [DeepSpeed](https://github.com/microsoft/DeepSpeed), [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/HEAD/applications/DeepSpeed-Chat) and [Safe-RLHF](https://github.com/PKU-Alignment/safe-rlhf).

Thanks for their wonderful works and their efforts to further promote LLM research.
Stream Aligner and its related assets are built and open-sourced with love and respect ❤️.

This work is supported and funded by the Institute of AI, Peking University.

<table width="50%" cellspacing="0" cellpadding="0">
  <tr align="center" valign="middle">
    <td width="40%">
      <a href="https://www.ai.pku.edu.cn/">
        <img src="assets/pku-ai.png" width="100%"/>
      </a>
    </td>
  </tr>
</table>

## Citation

Please cite our paper if you find this repository useful.

```bibtex
@inproceedings{lou2025stream,
    title={Stream Aligner: Efficient Sentence-Level Alignment via Distribution Induction},
    author={Hantao Lou and Jiaming Ji and Kaile Wang and Yaodong Yang},
    booktitle={The 39th Annual AAAI Conference on Artificial Intelligence},
    year={2025}
}
```
