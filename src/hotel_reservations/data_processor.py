import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup

class DataProcessor:
    def __init__(self, filepath, config):
        self.df = self.load_data(filepath)
        self.config = config
        self.train_table_uc = self.config["catalog_name"] + "." + self.config["schema_name"] + "." + "train_set"
        self.test_table_uc = self.config["catalog_name"] + "." + self.config["schema_name"] + "." + "test_set"
        self.function_name = self.config["catalog_name"] + "." + self.config["schema_name"] + "." + "calculate_house_age"

    def load_data(self, filepath):
        return pd.read_csv(filepath)   
    
    def split_data(self, test_size=0.2, random_state=42):
        # Remove rows with missing target
        target = self.config["target"]
        self.df = self.df.dropna(subset=[target])

        # Separate features and target
        X = self.df[self.config["num_features"] + self.config["cat_features"]]
        y = self.df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        train_set = pd.concat([X_train, y_train], axis=1)
        test_set = pd.concat([X_test, y_test], axis=1)

        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))   
        
        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))
        
        train_set_with_timestamp.write.mode("append").saveAsTable(self.train_table_uc)
        test_set_with_timestamp.write.mode("append").saveAsTable(self.test_table_uc)

        spark.sql(f"ALTER TABLE {self.train_table_uc} "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
        
        spark.sql(f"ALTER TABLE {self.test_table_uc} "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);") 

        train_set_spark = spark.sql(f'select * from {self.train_table_uc}')
        test_set_spark = spark.sql(f'select * from {self.test_table_uc}')  

        return train_set_spark, test_set_spark
    
    def create_feature_function(self, spark: SparkSession):        
        spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(NoWeekNights INT, NoWeekendNights INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        return NoWeekNights + NoWeekendNights
        $$
        """)

    def feature_engineering(self, train_set_spark, spark: SparkSession):
        self.create_feature_function(spark=spark)

        fe = feature_engineering.FeatureEngineeringClient()

        training_set = fe.create_training_set(
            df=train_set_spark.withColumn("no_of_week_nights", train_set_spark["no_of_week_nights"].cast("int")).withColumn("no_of_weekend_nights", train_set_spark["no_of_weekend_nights"].cast("int")),
            label=self.config["target"],
            feature_lookups=[
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="TotalNoWeekNights",
                    input_bindings={"NoWeekNights": "no_of_week_nights", "NoWeekendNights": "no_of_weekend_nights"},
                )
            ],
            exclude_columns=["update_timestamp_utc"]
        )

        train_set_spark = training_set.load_df()

        return training_set, train_set_spark
    
    def get_X_y_datasets(self, train_set_spark, test_set_spark, spark: SparkSession):
        """Read the train and test sets from Databricks tables."""

        train_set = train_set_spark.toPandas()
        test_set = test_set_spark.toPandas()

        X_train = train_set[self.config["num_features"] + self.config["cat_features"]]
        y_train = train_set[self.config["target"]]

        X_test = test_set[self.config["num_features"] + self.config["cat_features"]]
        y_test = test_set[self.config["target"]]

        return X_train, y_train, X_test, y_test

