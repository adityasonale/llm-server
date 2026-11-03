"""
Model Registry — promotes validated models into the MLflow registry.

Usage:
    registry = ModelRegistry(tracking_uri="sqlite:///mlflow.db")
    registry.register(
        model_name="local-minichat-1.5-3b",
        benchmark_run_id="abc123",
        eval_run_id="def456",
        metadata={"provider": "local", "model": "minichat-1.5-3b"}
    )
    registry.promote("local-minichat-1.5-3b", version=1, alias="champion")
"""

import mlflow
from mlflow.pyfunc import PythonModel
from mlflow import MlflowClient


class ModelRegistry:
    def __init__(self, tracking_uri: str, experiment_name: str = "Model Registry"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._client = MlflowClient()

    def register(
        self,
        model_name: str,
        python_model: PythonModel | None = None,
        benchmark_run_id: str | None = None,
        eval_run_id: str | None = None,
        active_run_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        if active_run_id:
            # reuse the existing run, no nesting
            try:
                self._client.create_registered_model(model_name)
            except mlflow.exceptions.MlflowException:
                pass
            self._client.create_model_version(
                name=model_name,
                source="",
                run_id=active_run_id,
            )
            version = self._client.get_latest_versions(model_name)[0].version
        else:
            # standalone registration with its own run
            with mlflow.start_run() as active_run:
                mlflow.set_tag("mlflow.runName", f"register-{model_name}")
                if metadata:
                    for k, v in metadata.items():
                        mlflow.log_param(k, v)
                if python_model is not None:
                    mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=python_model,
                        registered_model_name=model_name,
                    )
                else:
                    try:
                        self._client.create_registered_model(model_name)
                    except mlflow.exceptions.MlflowException:
                        pass
                    self._client.create_model_version(
                        name=model_name,
                        source="",
                        run_id=active_run.info.run_id,
                    )
            version = self._client.get_latest_versions(model_name)[0].version

        if benchmark_run_id:
            self._client.set_model_version_tag(model_name, version, "benchmark_run_id", benchmark_run_id)
        if eval_run_id:
            self._client.set_model_version_tag(model_name, version, "eval_run_id", eval_run_id)

        print(f"Registered {model_name} v{version}")
        return version

    def promote(self, model_name: str, version: str | int, alias: str) -> None:
        """
        Assign an alias to a version. e.g. promote("local-minichat-1.5-3b", 2, "champion")
        Moves the alias from whatever version it was on before.
        """
        self._client.set_registered_model_alias(model_name, alias, str(version))
        print(f"{model_name} v{version} → {alias}")

    def demote(self, model_name: str, alias: str) -> None:
        """Remove an alias from whatever version it currently points to."""
        self._client.delete_registered_model_alias(model_name, alias)
        print(f"Removed alias '{alias}' from {model_name}")

    def get_champion(self, model_name: str) -> dict:
        """Returns info about the current champion version."""
        version = self._client.get_model_version_by_alias(model_name, "champion")
        return {
            "version": version.version,
            "benchmark_run_id": version.tags.get("benchmark_run_id"),
            "eval_run_id": version.tags.get("eval_run_id"),
        }

    def list_versions(self, model_name: str) -> list[dict]:
        """List all versions and their aliases for a model."""
        versions = self._client.get_latest_versions(model_name)
        return [
            {
                "version": v.version,
                "aliases": self._client.get_model_version(model_name, v.version).aliases,
                "tags": v.tags,
            }
            for v in versions
        ]