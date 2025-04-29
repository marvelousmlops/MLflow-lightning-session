import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

class ModelServing:
    def __init__(self, model_name: str, endpoint_name: str):
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name
        self.model_name = model_name

    def get_latest_model_version(self):
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        print(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(self, version: str = "latest", workload_size: str = "Small", scale_to_zero: bool = True):
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        if version == "latest":
            entity_version = self.get_latest_model_version()
        else:
            entity_version = version

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
                environment_vars={
                    "OPENAI_API_KEY": "{{secrets/mlflow_genai/OPENAI_API_KEY}}",
                    "GEMINI_API_KEY": "{{secrets/mlflow_genai/GEMINI_API_KEY}}"
                    }
            )
        ]

        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
        else:
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)
