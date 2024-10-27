import logging
import yaml
import json
import mlflow
from pyspark.sql import SparkSession

# import subprocess
# import sys
# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# package = "/Volumes/mdl_europe_anz_dev/patrick_mlops/mlops_course/mlops_with_databricks-0.0.1-py3-none-any.whl"
# install(package)

from src.hotel_reservations.data_processor import DataProcessor

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

# Preprocess the data
data_processor.preprocess_data()
logger.info("Data preprocessed.")

# Split Train and Test data
train_set, test_set = data_processor.split_data()
logger.info("Train and Test data splitted.")

# Save data to catalog
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
logger.info("Data saved to catalog.")

# Create mlflow experiment
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/Hotel_Reservations")
mlflow.set_experiment_tags({"repository_name": "Patrick_MLOps"})

experiments = mlflow.search_experiments(
    filter_string="tags.repository_name='Patrick_MLOps'"
)

print(experiments)

# Write mlflow experiment to json file
with open("mlflow_experiment.json", "w") as json_file:
    json.dump(experiments[0].__dict__, json_file, indent=4)

# Start MLFlow run
with mlflow.start_run(
    run_name="demo_run",
    tags={"git_sha": "ffa63b430205ff7",
          "branch": "week2"},
    description="demo run",
) as run:
    mlflow.log_params({"type": "demo"})
    mlflow.log_metrics({"metric1": 1.0, "metric2": 2.0})

# Get Run info
run_id = mlflow.search_runs(
    experiment_names=["/Shared/Hotel_Reservations"],
    filter_string="tags.git_sha='ffa63b430205ff7'",
).run_id[0]

run_info = mlflow.get_run(run_id=f"{run_id}").to_dictionary()

print(run_info)
print(run_info["data"]["metrics"])
print(run_info["data"]["params"])

# Write run info to json file
with open("run_info.json", "w") as json_file:
    json.dump(run_info, json_file, indent=4)

