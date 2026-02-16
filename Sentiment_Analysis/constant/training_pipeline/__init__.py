import os, sys
import pandas as pd
import numpy as np

"""Defining common constant variable for training pipeline"""
TARGET_COLUMN = "rating"
PIPELINE_NAME = "SentimentAnalysis"
ARTIFACT_DIR = "Artifacts"
FILE_NAME = "all_kindle_review .csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

# SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

# SAVED_MODEL_DIR = os.path.join("saved_models")
# MODEL_FILE_NAME = "model.pkl"

# """Data Ingestion related constant start with DATA_INGESTION VAR NAME """

DATA_INGESTION_COLLECTION_NAME = "Sentiment_Analysis"
DATA_INGESTION_DATABASE_NAME = "Iamlati"
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION = 0.2

# """ Data Validation related constant start with DATA_VALIDATION VAR NAME """
# DATA_VALIDATION_DIR_NAME = "data_validation"
# DATA_VALIDATION_VALID_DIR = "validated"
# DATA_VALIDATION_INVALID_DIR = "invalid"
# DATA_VALIDATION_DRIFT_REPORT_DIR = "drift_report"
# DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = "report.yaml"
# PREPROCESSING_OBJECT_FILE_NAME:str="preprocessing.pkl"

# """ Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME """
# #KKN IMPUTER TO REPLACE NAN VALUES
# DATA_TRANSFORMATION_DIR_NAME = "data_transformation"
# DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR = "transformed"
# DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR = "transformed_object"
# DATA_TRANSFORMATION_IMPUTER_PARAMS = {
#     "missing_values": np.nan,
#     "n_neighbors":3,
#     "weights": "uniform"
# }


# """
# Model Trainer related constant start with MODEL TRAINER VAR NAME
# """

# MODEL_TRAINER_DIR_NAME:str="model_trainer"
# MODEL_TRAINER_TRAINED_MODEL_DIR: str="trained_model"
# MODEL_TRAINER_TRAINED_MODEL_NAME: str="model.pkl"
# MODEL_TRAINER_EXPECTED_SCORE:float=0.6
# MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD:float =0.5

# TRAINING_BUCKET_NAME = "networksecurityone1"