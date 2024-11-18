import hashlib
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import pandas as pd
import requests
from databricks import feature_engineering
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from lightgbm import LGBMRegressor
from mlflow import MlflowClient
from mlflow.deployments import get_deploy_client
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_reservations.utils import adjust_predictions

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")  # It must be -uc for registering models to Unity Catalog
deploy_client = get_deploy_client("databricks")
mlflow_client = MlflowClient()


class MLFlowProcessor:
    def __init__(
        self, config, train_set_spark, test_set_spark, X_train, y_train, X_test, y_test, model_name, host, token
    ):
        self.config = config
        self.train_set_spark = train_set_spark
        self.test_set_spark = test_set_spark
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.full_model_path = self.config["catalog_name"] + "." + self.config["schema_name"] + "." + self.model_name
        self.host = host
        self.token = token

        try:
            self.workspace = WorkspaceClient()
        except Exception:
            self.workspace = WorkspaceClient(host=self.host, token=self.token)

    def preprocess_data(self, parameters):
        # Create preprocessing steps for numeric and categorical data
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config["num_features"]),
                ("cat", categorical_transformer, self.config["cat_features"]),
            ]
        )

        self.model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LGBMRegressor(**parameters))])

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def model_wrapper(self, X_test):
        class HousePriceModelWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, model):
                self.model = model

            def predict(self, context, model_input):
                if isinstance(model_input, pd.DataFrame):
                    predictions = self.model.predict(model_input)
                    predictions = {"Prediction": adjust_predictions(predictions[0])}
                    return predictions
                else:
                    raise ValueError("Input must be a pandas DataFrame.")

        wrapped_model = HousePriceModelWrapper(self.model)
        example_prediction = wrapped_model.predict(context=None, model_input=X_test.iloc[0:1])

        return wrapped_model, example_prediction

    def model_wrapper_ab_test(self, models, X_test):
        class HousePriceModelWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, model):
                self.model = model
                self.model_a = models[0]
                self.model_b = models[1]

            def predict(self, context, model_input):
                if isinstance(model_input, pd.DataFrame):
                    house_id = str(model_input["Booking_ID"].str[-5:].values[0])
                    hashed_id = hashlib.md5(house_id.encode(encoding="UTF-8")).hexdigest()
                    # convert a hexadecimal (base-16) string into an integer
                    if int(hashed_id, 16) % 2:
                        predictions = self.model_a.predict(model_input)
                        return {"Prediction": predictions[0], "model": "Model A"}
                    else:
                        predictions = self.model_b.predict(model_input)
                        return {"Prediction": predictions[0], "model": "Model B"}
                else:
                    raise ValueError("Input must be a pandas DataFrame.")

        wrapped_model = HousePriceModelWrapper(models)
        example_prediction = wrapped_model.predict(context=None, model_input=X_test.iloc[0:1])

        return wrapped_model, example_prediction

    def evaluate(self, parameters):
        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R2 Score: {r2}")

        # Log parameters, metrics, and the model to MLflow
        mlflow.log_params(parameters)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

    def log_model(self, artifact_path):
        signature = infer_signature(model_input=self.X_train, model_output=self.y_pred)

        dataset = mlflow.data.from_spark(
            self.train_set_spark,
            table_name=self.config["catalog_name"] + "." + self.config["schema_name"] + "." + "train_set",
            version="0",
        )
        mlflow.log_input(dataset, context="training")

        mlflow.sklearn.log_model(sk_model=self.model, artifact_path=artifact_path, signature=signature)

    def log_model_fe(self, training_set, artifact_path):
        signature = infer_signature(model_input=self.X_train, model_output=self.y_pred)

        dataset = mlflow.data.from_spark(
            self.train_set_spark,
            table_name=self.config["catalog_name"] + "." + self.config["schema_name"] + "." + "train_set",
            version="0",
        )
        mlflow.log_input(dataset, context="training")

        fe = feature_engineering.FeatureEngineeringClient()

        fe.log_model(
            model=self.model,
            flavor=mlflow.sklearn,
            artifact_path=artifact_path,
            training_set=training_set,
            signature=signature,
        )

    def log_model_custom(self, artifact_path, wrapped_model, example_prediction):
        signature = infer_signature(model_input=self.X_train, model_output={"Prediction": example_prediction})

        dataset = mlflow.data.from_spark(
            self.train_set_spark,
            table_name=self.config["catalog_name"] + "." + self.config["schema_name"] + "." + "train_set",
            version="0",
        )
        mlflow.log_input(dataset, context="training")

        mlflow.pyfunc.log_model(
            python_model=wrapped_model,
            artifact_path=artifact_path,
            code_paths=[
                "/Volumes/mdl_europe_anz_dev/patrick_mlops/mlops_course/mlops_with_databricks-0.0.1-py3-none-any.whl"
            ],
            signature=signature,
        )

    def register_model(self, git_sha, model_version_alias, artifact_path, experiment_name):
        run_id = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=f"tags.git_sha='{git_sha}'",
        ).run_id[0]

        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/{artifact_path}", name=self.full_model_path, tags={"git_sha": f"{git_sha}"}
        )

        mlflow_client.set_registered_model_alias(self.full_model_path, model_version_alias, model_version.version)

        return run_id

    def load_model(self, model_version_alias):
        loaded_model = mlflow.pyfunc.load_model(f"models:/{self.full_model_path}@{model_version_alias}")

        return loaded_model

    def load_custom_model(self, run_id, artifact_path):
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{artifact_path}")
        loaded_model.unwrap_python_model()

        return loaded_model

    def load_dataset_from_model(self, run_id):
        run = mlflow.get_run(run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)

        return dataset_source

    def get_model_version_by_alias(self, model_version_alias):
        model_version_by_alias = mlflow_client.get_model_version_by_alias(self.full_model_path, model_version_alias)

        return model_version_by_alias

    def get_model_version_by_tag(self, git_sha):
        filter_string = f"name='{self.full_model_path}' and tags.git_sha='{git_sha}'"
        model_version_by_tag = mlflow_client.search_model_versions(filter_string)

        return model_version_by_tag

    def create_online_table(self):
        online_table_name = self.config["catalog_name"] + "." + self.config["schema_name"] + "." + "fe_online"
        spec = OnlineTableSpec(
            primary_key_columns=["Booking_ID"],
            source_table_full_name=self.config["catalog_name"] + "." + self.config["schema_name"] + "." + "fe_table",
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
            perform_full_copy=False,
        )

        self.workspace.online_tables.create(name=online_table_name, spec=spec)

    def create_model_serving_endpoint(self, model_serving_name, model_version):
        deploy_client.create_endpoint(
            name=model_serving_name,
            config={
                "served_entities": [
                    {
                        "entity_name": self.full_model_path,
                        "entity_version": "1",
                        "workload_size": "Small",
                        "scale_to_zero_enabled": True,
                    }
                ],
                "traffic_config": {
                    "routes": [{"served_model_name": f"{self.model_name}-1", "traffic_percentage": 100}]
                },
            },
        )

    def call_model_serving_endpoint(self, train_set, model_serving_name):
        required_columns = self.config["num_features"] + self.config["cat_features"]
        sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
        dataframe_records = [[record] for record in sampled_records]

        start_time = time.time()

        model_serving_endpoint = f"https://{self.host}/serving-endpoints/{model_serving_name}/invocations"
        response = requests.post(
            f"{model_serving_endpoint}",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"dataframe_records": dataframe_records[0]},
        )

        end_time = time.time()
        execution_time = end_time - start_time

        print("Response status:", response.status_code)
        print("Reponse text:", response.text)
        print("Execution time:", execution_time, "seconds")

    def model_serving_loadtest(self, train_set, model_serving_name, num_requests):
        required_columns = self.config["num_features"] + self.config["cat_features"]
        sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
        dataframe_records = [[record] for record in sampled_records]

        model_serving_endpoint = f"https://{self.host}/serving-endpoints/{model_serving_name}/invocations"

        # Function to make a request and record latency
        def send_request():
            random_record = random.choice(dataframe_records)
            start_time = time.time()
            response = requests.post(
                model_serving_endpoint,
                headers={"Authorization": f"Bearer {self.token}"},
                json={"dataframe_records": random_record},
            )
            end_time = time.time()
            latency = end_time - start_time
            return response.status_code, latency

        total_start_time = time.time()
        latencies = []

        # Send requests concurrently
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(send_request) for _ in range(num_requests)]

            for future in as_completed(futures):
                status_code, latency = future.result()
                latencies.append(latency)

        total_end_time = time.time()
        total_execution_time = total_end_time - total_start_time

        # Calculate the average latency
        average_latency = sum(latencies) / len(latencies)

        print("\nTotal execution time:", total_execution_time, "seconds")
        print("Average latency per request:", average_latency, "seconds")
