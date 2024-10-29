import logging
import yaml
import json
import mlflow
from pyspark.sql import SparkSession
import subprocess
import sys

# for package in ["/Volumes/mdl_europe_anz_dev/patrick_mlops/mlops_course/mlops_with_databricks-0.0.1-py3-none-any.whl"]:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# dbutils.library.restartPython()

from hotel_reservations.data_processor import DataProcessor
from hotel_reservations.mlflow_processor import MLFlowProcessor

spark = SparkSession.builder.getOrCreate()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load configuration
with open("project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# Initialize DataProcessor
data_processor = DataProcessor("/Volumes/mdl_europe_anz_dev/patrick_mlops/mlops_course/hotel_reservations.csv", config)
logger.info("DataProcessor initialized.")

# Split Train and Test data
train_set, test_set = data_processor.split_data()
logger.info("Train and Test data splitted.")

# Save data to catalog
train_set_spark, test_set_spark = data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
logger.info("Data saved to catalog.")

# Feature Engineering
training_set, train_set_spark = data_processor.feature_engineering(train_set_spark, spark=spark)
logger.info("Feature Engineering completed.")

# Split into X an y datasets
X_train, y_train, X_test, y_test = data_processor.get_X_y_datasets(train_set_spark, test_set_spark, spark=spark, func_features = config["func_features"] )
logger.info("Data read from catalog.")

# Initialize MLFlow Processor
model = MLFlowProcessor(config, train_set_spark, test_set_spark, X_train, y_train, X_test, y_test)
logger.info("MLFlow Processor initialized.")

# Create preprocessing steps and pipeline
model.preprocess_data()
logger.info("Pipeline created")

# Start an MLflow run to track the training process
mlflow.set_experiment(experiment_name=config["experiment_name"])
mlflow.set_experiment_tags({"repository_name": config["repository_name"]})
git_sha = "ffa63b430205ff7"

with mlflow.start_run(
    tags={"git_sha": f"{git_sha}",
        "branch": config["branch"]},
) as run:
    run_id = run.info.run_id

    # Train model and create MLFlow experiment
    model.train()
    logger.info("Model training and MLFlow experiment created.")

    # Create predictions
    model.predict()
    logger.info("Model predictions created.")

    # Evaluate model and log metrics to MLFlow experiment
    model.evaluate()
    logger.info("Model evaluated and logged in MLFlow experiment.")

    # Log model with feature engineering to MLFlow experiment
    model.log_model_fe(training_set)
    logger.info("Model logged to MLFlow experiment.")

    # Register model to MLFlow
    run_id = model.register_model(git_sha)
    logger.info("Model register to MLFlow.")

# Load registered model
loaded_model = model.load_model()

# Load dataset from registered model
dataset_source = model.load_dataset_from_model(run_id)
dataset_source.load()
logger.info("Dataset loaded from registered model.")

# Get model version by alias
model_version_by_alias = model.get_model_version_by_alias()
logger.info("Model version by alias loaded.")