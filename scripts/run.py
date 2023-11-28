import uuid
from typing import List

from pydantic import BaseModel, Field

from medplexity.chains.chain import Chain
from medplexity.chains.vqa_chain import VQAChain
from medplexity.llms.llm import LLM
from medplexity.llms.openai_caller import OpenAI
from medplexity.benchmarks.dataset_factory import DatasetFactory
from medplexity.medharness import Medharness
from medplexity.prompts.vqa_prompt import VQAPromptTemplate
from medplexity.storage.supabase_client import SupabaseEvaluationSaver

from datetime import datetime


DATASET = "vqarad"
SPLIT_TYPE = "test"

prompt = VQAPromptTemplate()

llms: List[LLM] = [
    OpenAI(
        model="gpt-4-vision-preview",
        temperature=0,
    ),
    # OpenAI(
    #     model="gpt-4-1106-preview",
    #     temperature=0,
    # ),
    # Deepinfra(
    #     model="llama-2-70b-chat-hf",
    #     temperature=0.01,
    # ),
    # Deepinfra(
    #     model="mistral-7b-instruct",
    #     temperature=0.01,
    # ),
]


class RunConfig(BaseModel):
    chain: Chain
    model: str
    dataset: str
    split_type: str
    prompt_template: str
    experiment_id: uuid.UUID = Field(default_factory=uuid.uuid4)

    K: int = 20
    ignore_errors: bool = True

    class Config:
        arbitrary_types_allowed = True


chains: List[Chain] = []

for llm in llms:
    chains.append(
        VQAChain(
            llm=llm,
        )
    )


run_configs: List[RunConfig] = []
for chain in chains:
    run_configs.append(
        RunConfig(
            chain=chain,
            model=chain.llm.model,
            dataset=DATASET,
            split_type=SPLIT_TYPE,
            prompt_template=prompt.PROMPT,
        )
    )


def run(config: RunConfig):
    print(f"Running evaluation for config: {config}")

    harness = Medharness(
        dataset=DatasetFactory().build(config.dataset, config.split_type),
        chain=config.chain,
    )

    print("Evaluating:")
    harness.run(k=config.K, ignore_errors=True)

    print("Saving results to file")
    evaluation_file_path = (
        f"{config.model}-{config.dataset}-{config.split_type}-{config.experiment_id}"
    )

    harness.save_results(
        evaluation_file_path,
        additional_data={
            "prompt_template": config.prompt_template,
            "date": datetime.now().date().strftime("%d-%m-%Y"),
        },
    )

    print("Saving to persistent storage")
    evaluation_saver = SupabaseEvaluationSaver()

    evaluation_saver.save_evaluation(
        file_name=evaluation_file_path,
        model=config.model,
        benchmark_id=config.dataset,
        split_type=config.split_type,
    )


if __name__ == "__main__":
    for config in run_configs:
        run(config)
