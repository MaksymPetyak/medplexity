"""Classes herer are meant just to upload the results to Supabase for visualisations on medplexityai.com."""

import os
import uuid
from pathlib import Path
import supabase
from dotenv import load_dotenv

load_dotenv()


class SupabaseClient:
    """Supabase client to interact with the database and store results"""

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
    ):
        if supabase_url is None:
            supabase_url = os.environ.get("SUPABASE_URL")
        if supabase_key is None:
            supabase_key = os.environ.get("SUPABASE_KEY")

        self._client = supabase.create_client(supabase_url, supabase_key)

    def upload_file(
        self,
        filepath: Path,
        bucket_name: str,
        path_on_storage: str,
        file_options: dict | None = None,
    ) -> str:
        with open(filepath, "rb") as f:
            response = self._client.storage.from_(bucket_name).upload(
                file=f, path=path_on_storage, file_options=file_options
            )
            response_json = response.json()
            return (
                "https://fqmgogeamrlfnziacygu.supabase.co/storage/v1/object/public/"
                + response_json["Key"]
            )

    def insert(self, table_name: str, values: dict):
        self._client.table(table_name).insert(values).execute()

    def update_entry(self, table_name: str, id: str, values: dict):
        self._client.table(table_name).update(values).eq("id", id).execute()

    def check_id_exists(self, table_name: str, id: str) -> bool:
        result = self._client.table(table_name).select("id").eq("id", id).execute()
        return len(result.data) > 0


class TableWrapper:
    table_name = ""

    def __init__(self, client: SupabaseClient):
        self.client = client

    def check_id_exists(self, id: str) -> bool:
        return self.client.check_id_exists(self.table_name, id)


class BenchmarkDB(TableWrapper):
    table_name = "benchmarks"

    def insert(self, values: dict):
        self.client.insert(self.table_name, values)


class DatasetConfigDB(TableWrapper):
    table_name = "dataset_configs"

    def insert(self, id: str, benchmark: str, split_type: str, subtype: str | None):
        values = {
            "id": id,
            "benchmark": benchmark,
            "split_type": split_type,
            "subtype": subtype,
        }
        self.client.insert(self.table_name, values)

    def _get_configurations(
        self, benchmark_id: str, split_type: str, subtype: str | None
    ):
        query = (
            self.client._client.table(self.table_name)
            .select("*")
            .eq("benchmark", benchmark_id)
            .eq("split_type", split_type)
        )

        # Need to handle NULL values accordingly
        if subtype is None:
            query = query.is_("subtype", "null")
        else:
            query = query.eq("subtype", subtype)

        existing_configs = query.execute()

        return existing_configs.data

    def dataset_configuration_exists(
        self, benchmark_id: str, split_type: str, subtype: str | None
    ) -> bool:
        return len(self._get_configurations(benchmark_id, split_type, subtype)) > 0

    def get_configuration_id(
        self, benchmark_id: str, split_type: str, subtype: str | None
    ) -> str:
        data = self._get_configurations(benchmark_id, split_type, subtype)
        return data[0]["id"]


class EvaluationsDB(TableWrapper):
    table_name = "evaluations"

    def insert(self, dataset_config: str, model: str, evaluation_url: str):
        values = {
            "dataset_config": dataset_config,
            "model": model,
            "evaluation_url": evaluation_url,
        }
        self.client.insert("evaluations", values)

    def update_eval(self, id: str, evaluation_url: str):
        self.client.update_entry("evaluations", id, {"evaluation_url": evaluation_url})

    def _get_evals(self, dataset_config_id: str, model: str):
        query = (
            self.client._client.table(self.table_name)
            .select("*")
            .eq("dataset_config", dataset_config_id)
            .eq("model", model)
        )
        existing_configs = query.execute()

        return existing_configs.data

    def eval_exists(self, dataset_config_id: str, model: str) -> bool:
        return len(self._get_evals(dataset_config_id, model)) > 0

    def get_existing_eval_id(self, dataset_config_id: str, model: str) -> str:
        data = self._get_evals(dataset_config_id, model)
        return data[0]["id"]


class SupabaseEvaluationSaver:
    BUCKET_NAME = "EvalRuns"

    def __init__(self):
        self.client = SupabaseClient()
        self.benchmark_db = BenchmarkDB(self.client)
        self.dataset_config_db = DatasetConfigDB(self.client)
        self.evaluations_db = EvaluationsDB(self.client)

    def save_evaluation(
        self,
        file_name: str,
        model: str,
        benchmark_id: str,
        split_type: str,
        subtype: str | None = None,
        new_benchmark_config: dict | None = None,
    ):
        if not self.benchmark_db.check_id_exists(benchmark_id) and new_benchmark_config:
            if new_benchmark_config:
                self.benchmark_db.insert(new_benchmark_config)
            else:
                raise ValueError("No Benchmark ID or new benchmark config provided")

        dataset_config_id = None
        if not self.dataset_config_db.dataset_configuration_exists(
            benchmark_id, split_type, subtype
        ):
            dataset_config_id = str(uuid.uuid4())
            self.dataset_config_db.insert(
                dataset_config_id, benchmark_id, split_type, subtype
            )
        else:
            dataset_config_id = self.dataset_config_db.get_configuration_id(
                benchmark_id, split_type, subtype
            )

        file_path = Path(file_name)
        path_on_storage = f"{file_path.name}"
        evaluation_url = self.client.upload_file(
            file_path, self.BUCKET_NAME, path_on_storage
        )

        if self.evaluations_db.eval_exists(dataset_config_id, model):
            eval_id = self.evaluations_db.get_existing_eval_id(dataset_config_id, model)

            self.evaluations_db.update_eval(eval_id, evaluation_url)
        else:
            self.evaluations_db.insert(dataset_config_id, model, evaluation_url)
