## Medplexity 

![Release](https://img.shields.io/pypi/v/medplexity?label=Release&style=flat-square)
[![Documentation Status](https://readthedocs.org/projects/medplexity/badge/?version=latest)](https://medplexity.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://dcbadge.vercel.app/api/server/jUKkgqVzQ?style=flat&compact=true)](https://discord.gg/Wr2RzVgdE)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Open in Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/MaksymPetyak/medplexity/blob/main/notebooks/Getting%20started.ipynb)

<p align="center">
  <a href="https://www.medplexityai.com/">Medplexity explorer</a> •
  <a href="https://github.com/MaksymPetyak/medplexity-frontend">Frontend GitHub repository</a> •
  <a href="https://medplexity.substack.com/">Substack</a>
</p>


Medplexity is a python library to help with evaluation of LLMs for medical applications.

<img src="images/medplexity-logo.png" alt="medplexity-logo" width="512px" style="border-radius: 16px;"/>

It is designed to help with the following tasks:
- Evaluating performance of LLMs on existing medical datasets and benchmarks. E.g. MedQA, PubMedQA, etc.
- Comparing performance of different prompts, models, and architectures.
- Exporting results of evaluation for visualisation and further analysis. 

The goal is to help answer questions like "How much better would GPT-4 perform given a vector database to load certain resources?".


## 🔧 Quick install
```bash
pip install medplexity
```

## 📖 Documentation

Documentation can be found [here](https://medplexity.readthedocs.io/en/latest/).


## Example
See our ["Getting Started" notebook](https://colab.research.google.com/github/MaksymPetyak/medplexity/blob/main/notebooks/Getting%20started.ipynb) for a full example with MedMCQA dataset.

## Contributions

Contributions are welcome! Check out the todos below, and feel free to open a pull request.
Remember to install `pre-commit` to be compliant with our standards:

```bash
pre-commit install
```

Feel free to raise any questions on [Discord](https://discord.gg/Wr2RzVgdE)

## Todos
Some initial todos include:
- [x] Multiple-Choice datasets
  - [x] Add MedMCQA dataset
  - [x] Add PubMedQA dataset
  - [x] Add MedQA dataset
  - [x] Add MMLU dataset
- [ ] Long-form question answering datasets
  - [x] Add HealthSearchQA dataset
  - [x] Add MedicationQA dataset
  - [ ] Add LiveQA dataset
- [ ] Explore datasets for multi-modality, specifically vision tasks for GPT-4V.
- [ ] LLMs
    - [x] Wrapper for OpenAI
    - [x] Wrapper for deepinfra
    - [ ] Wrapper for Google PALM
    - [ ] Wrapper for HuggingFace text-gen
- [x] Jupyter notebook quickstart
- [x] Example with langchain integration
- [x] Visualisation of results
  - [x] Add export of evaluations
  - [x] Frontend for exploring exported results


## Explorer
In addition to the library, we are also building a web app to explore the results of evaluations.
The explorer is available at [medplexityai.com](https://www.medplexityai.com/).
It's also open-sourced, see the [frontend repository](https://github.com/MaksymPetyak/medplexity-frontend).

## 📜 License
Medplexity is licensed under the MIT License. See the LICENSE file for more details.
