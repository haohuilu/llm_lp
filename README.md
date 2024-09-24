
# ComLLM: Enhancing Predictions of Disease Progression using Large Language Models

This repository contains the code and resources for the paper titled **"Can Large Language Models Enhance Predictions of Disease Progression? Investigating Through Disease Network Link Prediction"**, accepted at **EMNLP 2024**. The paper explores the application of Large Language Models (LLMs), such as GPT-4, in predicting disease progression through advanced graph prompting and Retrieval-Augmented Generation (RAG).

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Introduction

The project introduces ComLLM, a novel framework that leverages LLMs with graph prompting and RAG to predict disease comorbidity in disease networks. ComLLM consistently outperforms existing models, providing significant improvements in predicting disease progression across different network structures.

## Features

- **Graph Prompting**: Incorporates structural information from disease networks to improve link prediction.
- **Retrieval-Augmented Generation (RAG)**: Integrates external medical knowledge to enhance prediction accuracy.
- **Flexible Prompting Strategies**: Supports zero-shot, few-shot, and Chain-of-Thought (COT) prompting methods.
- **Multi-Model Compatibility**: Works with various LLMs, including GPT-4, GPT-3.5, LLaMA 2, LLaMA 3, and LLaMA 3.1.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/haohuilu/llm_lp.git
   cd llm_lp
   ```

2. Create a virtual environment and install the required dependencies:
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. Download the datasets:
   - The Human Disease Network (HDN) and Disease-Symptoms Network (DSN) datasets are required for training. Instructions for obtaining them can be found in the `data/` directory.


## Experiments

We conducted experiments on two main datasets:
- **Human Disease Network (HDN)**
- **Disease-Symptoms Network (DSN)**

### Reproducing Experiments
You can reproduce the experiments using the Jupyter notebooks provided in the `notebooks/` directory.

## Results

Our model, ComLLM, demonstrated significant improvements in predicting disease comorbidities:
- Achieved an average AUC improvement of 10.70% over baseline models in the Human Disease Network.
- Achieved an average AUC improvement of 6.07% in the Disease-Symptoms Network.

For more detailed results, refer to the `results/` directory.

## Citation

If you find this code helpful in your research, please cite our paper:

```bibtex
@inproceedings{lu2024comllm,
  title={Can Large Language Models Enhance Predictions of Disease Progression? Investigating Through Disease Network Link Prediction},
  author={Lu, Haohui and Naseem, Usman},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

