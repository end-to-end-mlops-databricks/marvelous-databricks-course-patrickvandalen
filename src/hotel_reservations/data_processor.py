import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataProcessor:
    def __init__(self, filepath, config):
        self.df = self.load_data(filepath)
        self.config = config
        # self.X = None
        # self.y = None
        # self.preprocessor = None
        self.train_table_uc = self.config["catalog_name"] + "." + self.config["schema_name"] + "." + "train_set"
        self.test_table_uc = self.config["catalog_name"] + "." + self.config["schema_name"] + "." + "test_set"

    def load_data(self, filepath):
        return pd.read_csv(filepath)

    def preprocess_data(self):
        # Remove rows with missing target
        target = self.config["target"]
        self.df = self.df.dropna(subset=[target])

        # Separate features and target
        self.X = self.df[self.config["num_features"] + self.config["cat_features"]]
        self.y = self.df[target]

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

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config["num_features"]),
                ("cat", categorical_transformer, self.config["cat_features"]),
            ]
        )

    def split_data(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
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
        
    def read_from_catalog(self, spark: SparkSession):
        """Read the train and test sets from Databricks tables."""

        train_set_spark = spark.sql(f'select * from {self.train_table_uc}')
        test_set_spark = spark.sql(f'select * from {self.test_table_uc}')

        train_set = train_set_spark.toPandas()
        test_set = test_set_spark.toPandas()

        X_train = train_set[self.config["num_features"] + self.config["cat_features"]]
        y_train = train_set[self.config["target"]]

        X_test = test_set[self.config["num_features"] + self.config["cat_features"]]
        y_test = test_set[self.config["target"]]

        return train_set_spark, test_set_spark, X_train, y_train, X_test, y_test