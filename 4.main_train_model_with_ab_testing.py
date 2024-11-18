import logging

import mlflow
import yaml
from pyspark.sql import SparkSession

# import subprocess
# import sys
# for package in ["/Volumes/mdl_europe_anz_dev/patrick_mlops/mlops_course/mlops_with_databricks-0.0.1-py3-none-any.whl"]:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# dbutils.library.restartPython()
from hotel_reservations.data_processor import DataProcessor
from hotel_reservations.mlflow_processor import MLFlowProcessor

spark = SparkSession.builder.getOrCreate()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_dbutils(spark):
    try:
        from pyspark.dbutils import DBUtils

        dbutils = DBUtils(spark)
    except ImportError:
        import IPython

        dbutils = IPython.get_ipython().user_ns["dbutils"]
    return dbutils


dbutils = get_dbutils(spark)

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Load configuration
with open("project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# Initialize DataProcessor
data_processor = DataProcessor(
    "/Volumes/"
    + config["catalog_name"]
    + "/"
    + config["schema_name"]
    + "/"
    + config["volume_name"]
    + "/"
    + config["table_name"],
    config,
)
logger.info("DataProcessor initialized.")

# Split into Train and Test data
train_set, test_set = data_processor.split_data()
logger.info("Train and Test data splitted.")

# Save data to catalog
train_set_spark, test_set_spark = data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
logger.info("Data saved to catalog.")

# Split into X an y datasets
X_train, y_train, X_test, y_test = data_processor.get_X_y_datasets(train_set_spark, test_set_spark, spark=spark)
logger.info("Data read from catalog.")

# Initialize MLFlow Processor
model_name = "hotel_reservations_model_ab_testing"
model_serving_name = "hotel-reservations-model-serving-ab-testing"

model = MLFlowProcessor(
    config, train_set_spark, test_set_spark, X_train, y_train, X_test, y_test, model_name, host, token
)
logger.info("MLFlow Processor initialized.")

for ab_test_models in ["model_A", "model_B"]:
    if ab_test_models == "model_A":
        parameters = config["ab_test_parameters_a"]
    elif ab_test_models == "model_B":
        parameters = config["ab_test_parameters_b"]

    # Create preprocessing steps and pipeline
    model.preprocess_data(parameters)
    logger.info("Pipeline created")

    # Start an MLflow run to track the training process
    experiment_name = config["ab_test_experiment_name"]
    artifact_path = "lightgbm-pipeline-model"
    model_version_alias = ab_test_models
    git_sha = "ffa63b430205ff7"
    mlflow.set_experiment(experiment_name=experiment_name)
    mlflow.set_experiment_tags({"repository_name": config["repository_name"]})

    mlflow.start_run(
        tags={"model_class": ab_test_models, "git_sha": f"{git_sha}", "branch": config["branch"]},
    )

    run = mlflow.active_run()
    run_id = run.info.run_id

    # Train model and create MLFlow experiment
    model.train()
    logger.info("Model training and MLFlow experiment created.")

    # Create predictions
    model.predict()
    logger.info("Model predictions created.")

    # Evaluate model and log metrics to MLFlow experiment
    model.evaluate(parameters)
    logger.info("Model evaluated and logged in MLFlow experiment.")

    # Log model to MLFlow experiment
    model.log_model(artifact_path)
    logger.info("Model logged to MLFlow experiment.")

    # Register model to MLFlow
    run_id = model.register_model(git_sha, model_version_alias, artifact_path, experiment_name)
    logger.info("Model register to MLFlow.")

    mlflow.end_run()

# Load registered model A
model_version_alias = "model_A"
model_A = model.load_model(model_version_alias)
logger.info("Loaded Model A.")

# Load registered model B
model_version_alias = "model_B"
model_B = model.load_model(model_version_alias)
logger.info("Loaded Model B.")

# Wrap models A and B using hash for split
models = [model_A, model_B]
wrapped_model, example_prediction = model.model_wrapper_ab_test(models, X_test)

# Initialize MLFlow Processor
model_name = "hotel_reservations_model_ab_testing_wrapped"
model = MLFlowProcessor(
    config, train_set_spark, test_set_spark, X_train, y_train, X_test, y_test, model_name, host, token
)
logger.info("MLFlow Processor initialized.")

# Start an MLflow run to log and register the wrapped model
experiment_name = config["ab_test_experiment_name"]
artifact_path = "pyfunc-house-price-model-ab"
model_version_alias = "wrapped_model"
git_sha = "ffa63b430205ff7"
mlflow.set_experiment(experiment_name=experiment_name)
mlflow.set_experiment_tags({"repository_name": config["repository_name"]})

with mlflow.start_run(
    tags={"git_sha": f"{git_sha}", "branch": config["branch"]},
) as run:
    run_id = run.info.run_id

    # Log model to MLFlow experiment
    model.log_model_custom(artifact_path, wrapped_model, example_prediction)
    logger.info("Model logged to MLFlow experiment.")

    # Register model to MLFlow
    run_id = model.register_model(git_sha, model_version_alias, artifact_path, experiment_name)
    logger.info("Model register to MLFlow.")

# Load dataset from registered model
dataset_source = model.load_dataset_from_model(run_id)
dataset_source.load()
logger.info("Dataset loaded from registered model.")

# Get model version by alias
model_version = model.get_model_version_by_alias(model_version_alias)
logger.info("Model version by alias loaded.")

# Create Model Serving Endpoint
model.create_model_serving_endpoint(model_serving_name, model_version.version)
logger.info("Model serving endpoint created.")

# Call Model Serving Endpoint
model.call_model_serving_endpoint(train_set, model_serving_name)
logger.info("Model serving endpoint called.")
