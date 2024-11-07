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
model_name = "hotel_reservations_model_custom"
experiment_name = config["experiment_name"]
artifact_path = "lightgbm-pipeline-model"
model_version_alias = "the_best_model"
git_sha = "ffa63b430205ff7"
mlflow.set_experiment(experiment_name=experiment_name)
mlflow.set_experiment_tags({"repository_name": config["repository_name"]})

model = MLFlowProcessor(
    config, train_set_spark, test_set_spark, X_train, y_train, X_test, y_test, model_name, host, token
)
logger.info("MLFlow Processor initialized.")

# Create preprocessing steps and pipeline
model.preprocess_data(config["parameters"])
logger.info("Pipeline created")

# Start an MLflow run to track the training process
with mlflow.start_run(
    tags={"git_sha": f"{git_sha}", "branch": config["branch"]},
) as run:
    run_id = run.info.run_id

    # Train model and create MLFlow experiment
    model.train()
    logger.info("Model training and MLFlow experiment created.")

    # Create predictions
    model.predict()
    logger.info("Model predictions created.")

    # Wrap custom model
    wrapped_model, example_prediction = model.model_wrapper(X_test)
    logger.info("Model wrapped.")

    # Evaluate model and log metrics to MLFlow experiment
    model.evaluate(config["parameters"])
    logger.info("Model evaluated and logged in MLFlow experiment.")

    # Log model to MLFlow experiment
    model.log_model_custom(artifact_path, wrapped_model, example_prediction)
    logger.info("Model logged to MLFlow experiment.")

    # Register model to MLFlow
    run_id = model.register_model(git_sha, model_version_alias, artifact_path, experiment_name)
    logger.info("Model register to MLFlow.")

# Load custom model
loaded_model = model.load_custom_model(run_id, artifact_path)

# Load dataset from registered model
dataset_source = model.load_dataset_from_model(run_id)
dataset_source.load()
logger.info("Dataset loaded from registered model.")

# Get model version by alias
model_version_by_alias = model.get_model_version_by_alias(model_version_alias)
logger.info("Model version by alias loaded.")
