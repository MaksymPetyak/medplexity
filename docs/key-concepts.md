The goal of medplexity is to allow easy exploration of the capabilities of LLMs in the medical domain. 
To achieve this, we provide a set of tools to build chains with LLMs and then apply those chains to a dataset to generate a performance report. 
Let's go through these concepts one by one.

Note: see also [example notebook](https://github.com/MaksymPetyak/medplexity/blob/main/notebooks/MedQA.ipynb) for using OpenAI on MedQA dataset. 

## LLMs
`LLMs` (Large Language Models) are callable objects that take a string as input and return a string as output. 
They are the base that you can use in more complex chains that you create. 
We provide interfaces to use common APIs, for example OpenAI.

## Chains
To accommodate more intricate implementations and patterns, we aim to be very flexible on your main prediction function. 
For this purpose, we introduce the `Chain` class. 
This class represents a callable object designed to be a versatile wrapper around your LLM code. 
However, due to this flexibility, you might need to write custom code to tailor your chain for evaluation.
Generally Chains are expected to have LLMs as their base, and then implement additional logic, for examples, retrieval (RAG - Retrieval Augmented Generation) or agent patterns.
If you are familiar with Chains from Langchain, this is meant to be the same concept.


```python
from llms.openai_caller import OpenAI
from medplexity.chains.evaluation_adapter_chain import EvaluationAdapterChain

chain = EvaluationAdapterChain(
    llm=OpenAI(
        api_token=OPENAI_API_KEY
    ),
    input_adapter=input_adapter,
    output_adapter=output_adapter,
)
```

## Benchmarks
This is one of the main features of medplexity. 
We provide common interfaces to popular existing benchmarks in the medical space.
From any benchmark you can get a `Dataset`, which is an iterable of inputs and expected outputs.
You can then use this dataset for evaluation. 

```python
from medplexity.benchmarks.medqa.medqa_dataset_builder import MedQADatasetBuilder

# Get validation split of MedQA dataset.
dataset = MedQADatasetBuilder().build_dataset(split_type="validation")

# It's an iterable of Datapoints, where each datapoint has an input and an expected output, as well as any other metadata.
dataset[0]
```

## Evaluator
`Evaluators` glue everything together to generate a performance report.
You provide `Evaluators` with your `Chain` and then run it on a given `Dataset`.
After evaluation is complete you get an `EvaluationReport`, to look in detail at predictions on every datapoint and check where your chain is doing well and where it's not.