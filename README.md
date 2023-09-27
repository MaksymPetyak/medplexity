## Medplexity 

[![Documentation Status](https://readthedocs.org/projects/medplexity/badge/?version=latest)](https://medplexity.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://dcbadge.vercel.app/api/server/jUKkgqVzQ?style=flat&compact=true)](https://discord.gg/jUKkgqVzQ)


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
See [MedQA notebook](`notebooks/MedQA.ipynb`) for a full example with MedQA dataset.

## Contributions

Contributions are welcome! Check out the todos below, and feel free to open a pull request.
Remember to install `pre-commit` to be compliant with our standards:

```bash
pre-commit install
```

Feel free to raise any questions on [Discord](https://discord.gg/jUKkgqVzQ)

## Todos
Some initial todos include:
- [x] Multiple-Choice datasets
  - [x] Add MedMCQA dataset
  - [x] Add PubMedQA dataset
  - [x] Add MedQA dataset
  - [x] Add MMLU dataset
- [ ] Long-form question answering datasets
  - [x] Add HealthSearchQA dataset
  - [ ] Add MedicationQA dataset
  - [ ] Add LiveQA dataset
- [ ] LLMs
    - [x] Wrapper for OpenAI
    - [x] Wrapper for deepinfra
    - [ ] Wrapper for Google PALM
    - [ ] Wrapper for HuggingFace text-gen
- [ ] Jupyter notebook quickstart
- [ ] Example with langchain integration
- [ ] Visualisation of results
  - [ ] Add export of evaluations
  - [ ] Frontend for exploring exported results


## 📜 License
Medplexity is licensed under the MIT License. See the LICENSE file for more details.
