import mlflow
import time
import requests
import pandas as pd
from pyspark.sql import SparkSession
from databricks import feature_engineering
from lightgbm import LGBMRegressor
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route
)

from hotel_reservations.utils import adjust_predictions

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")  # It must be -uc for registering models to Unity Catalog
client = MlflowClient()
workspace = WorkspaceClient()

class MLFlowProcessor:
    def __init__(self, config, train_set_spark, test_set_spark, X_train, y_train, X_test, y_test, model_name):
        self.config = config
        self.train_set_spark = train_set_spark
        self.test_set_spark = test_set_spark
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name

    def preprocess_data(self):
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

        self.model = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", LGBMRegressor(**self.config["parameters"]))]
        )

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
                    predictions = self.model.predict(X_test)
                    predictions = {"Prediction": adjust_predictions(predictions[0])}
                    return predictions
                else:
                    raise ValueError("Input must be a pandas DataFrame.")

        self.wrapped_model = HousePriceModelWrapper(self.model)
        self.example_prediction = self.wrapped_model.predict(context=None, model_input=X_test.iloc[0:1])

    def evaluate(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R2 Score: {r2}")

        # Log parameters, metrics, and the model to MLflow
        mlflow.log_param("model_type", "LightGBM with preprocessing")
        mlflow.log_params(self.config["parameters"])
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

    def log_model(self):
        signature = infer_signature(model_input=self.X_train, model_output=self.y_pred)

        dataset = mlflow.data.from_spark(
            self.train_set_spark,
            table_name=self.config["catalog_name"] + "." + self.config["schema_name"] + "." + "train_set",
            version="0",
        )
        mlflow.log_input(dataset, context="training")

        mlflow.sklearn.log_model(sk_model=self.model, artifact_path="lightgbm-pipeline-model", signature=signature)

    def log_model_fe(self, training_set):
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
            artifact_path="lightgbm-pipeline-model",
            training_set=training_set,
            signature=signature,
        )

    def log_model_custom(self):
        signature = infer_signature(model_input=self.X_train, model_output={"Prediction": self.example_prediction})

        dataset = mlflow.data.from_spark(
            self.train_set_spark,
            table_name=self.config["catalog_name"] + "." + self.config["schema_name"] + "." + "train_set",
            version="0",
        )
        mlflow.log_input(dataset, context="training")

        mlflow.pyfunc.log_model(
            python_model=self.wrapped_model,
            artifact_path="lightgbm-pipeline-model",
            code_paths=[
                "/Volumes/mdl_europe_anz_dev/patrick_mlops/mlops_course/mlops_with_databricks-0.0.1-py3-none-any.whl"
            ],
            signature=signature,
        )

    def register_model(self, git_sha):
        run_id = mlflow.search_runs(
            experiment_names=[self.config["experiment_name"]],
            filter_string=f"tags.git_sha='{git_sha}'",
        ).run_id[0]

        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/lightgbm-pipeline-model", name=self.model_name, tags={"git_sha": f"{git_sha}"}
        )

        self.model_version_alias = "the_best_model"
        client.set_registered_model_alias(self.model_name, self.model_version_alias, model_version.version)

        return run_id

    def load_model(self):
        loaded_model = mlflow.pyfunc.load_model(f"models:/{self.model_name}@{self.model_version_alias}")

        return loaded_model

    def load_custom_model(self, run_id):
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")
        loaded_model.unwrap_python_model()

        return loaded_model

    def load_dataset_from_model(self, run_id):
        run = mlflow.get_run(run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)

        return dataset_source

    def get_model_version_by_alias(self):
        model_version_by_alias = client.get_model_version_by_alias(self.model_name, self.model_version_alias)

        return model_version_by_alias

    def get_model_version_by_tag(self, git_sha):
        filter_string = f"name='{self.model_name}' and tags.tag='{git_sha}'"
        model_version_by_tag = client.search_model_versions(filter_string)

        return model_version_by_tag
    
    def create_model_serving_endpoint(self, model_name, model_version):

        workspace.serving_endpoints.create(
            name="hotel-reservations-model-serving",
            config=EndpointCoreConfigInput(
                served_entities=[
                    ServedEntityInput(
                        entity_name=model_name,
                        scale_to_zero_enabled=True,
                        workload_size="Small",
                        entity_version=model_version,
                    )
                ],
            # Optional if only 1 entity is served
            traffic_config=TrafficConfig(
                routes=[
                    Route(served_model_name="hotel-reservations-model-serving-2",
                        traffic_percentage=100)
                ]
                ),
            ),
        )

    def call_model_serving_endpoint(self, train_set, token, host, spark: SparkSession):

        required_columns = self.config["num_features"] + self.config["cat_features"]
        print(required_columns)

        sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
        dataframe_records = [[record] for record in sampled_records]

        start_time = time.time()

        model_serving_endpoint = (
            f"https://{host}/serving-endpoints/hotel-reservations-model-serving/invocations"
        )
        response = requests.post(
            f"{model_serving_endpoint}",
            headers={"Authorization": f"Bearer {token}"},
            json={"dataframe_records": dataframe_records[0]},
        )

        end_time = time.time()
        execution_time = end_time - start_time

        print("Response status:", response.status_code)
        print("Reponse text:", response.text)
        print("Execution time:", execution_time, "seconds")
