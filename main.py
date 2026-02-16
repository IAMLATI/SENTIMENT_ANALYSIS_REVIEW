from Sentiment_Analysis.components.data_ingestion import DataIngestion
# from Sentiment_Analysis.components.data_validation import DataValidation
# from Sentiment_Analysis.components.data_transformation import DataTransformation
# from Sentiment_Analysis.components.model_trainer import ModelTrainer
from Sentiment_Analysis.exception.exception import SentimentAnalysisException
from Sentiment_Analysis.logging.logger import logging
from Sentiment_Analysis.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig

import sys

if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiated completed")
        print(dataingestionartifact)

        # data_validation_config = DataValidationConfig(trainingpipelineconfig)
        # data_validation = DataValidation(dataingestionartifact,data_validation_config)
        # logging.info("initiate the data validation")
        # data_validation_artifact=data_validation.initiate_data_validation()
        # logging.info(" data validation completed")
        # print(data_validation_artifact)

        # data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        # logging.info(" data Transformation Started")
        # data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        # data_transformation_artifact= data_transformation.initiate_data_transformation()
        # print(data_transformation_artifact)
        # logging.info(" data Transformation completed")

        # logging.info("Model Training started")
        # model_trainer_config=ModelTrainerConfig(trainingpipelineconfig)
        # model_trainer=ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        # model_trainer_artifact=model_trainer.initiate_model_trainer()

        # logging.info("mODEL TRAINING ARTIFACT CREATED")


    except Exception as e:
        raise SentimentAnalysisException(e, sys)