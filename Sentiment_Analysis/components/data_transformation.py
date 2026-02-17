import re

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
import sys, os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from Sentiment_Analysis.exception.exception import SentimentAnalysisException
from Sentiment_Analysis.logging.logger import logging
from Sentiment_Analysis.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from Sentiment_Analysis.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from Sentiment_Analysis.entity.config_entity import DataTransformationConfig
from Sentiment_Analysis.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise SentimentAnalysisException(e,sys)
        
    @staticmethod
    def read_data(file_path)-> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SentimentAnalysisException(e, sys)

    def _clean_text_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Receives a DataFrame with column 'reviewText' and returns a cleaned Series of strings.
        Used inside sklearn FunctionTransformer.
        """
        try:
            text_series = df["reviewText"].fillna("").astype(str)

            def lemmatizer_words(text: str) -> str:
                return " ".join([self._lemmatizer.lemmatize(word) for word in text.split()])

            def clean_text(x: str) -> str:
                x = str(x).lower()

                # remove urls/domains
                x = re.sub(
                    r"(https?://\S+|www\.\S+|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:/\S*)?)",
                    "",
                    x
                )

                # strip html
                x = BeautifulSoup(x, "lxml").get_text()

                # keep only alphanumeric + spaces
                x = re.sub(r"[^a-zA-Z0-9\s]", " ", x)

                # remove stopwords
                x = " ".join([w for w in x.split() if w not in self._stop_words])

                # normalize spaces
                x = " ".join(x.split())

                # lemmatize
                x = lemmatizer_words(x)

                return x

            return text_series.apply(clean_text)

        except Exception as e:
            raise SentimentAnalysisException(e, sys)

        
    def get_data_transformer_object(self) -> Pipeline:
        """
        reviewText -> cleaning -> CountVectorizer -> numeric feature matrix
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            text_cleaner = FunctionTransformer(self._clean_text_series, validate=False)

            processor = Pipeline(
                steps=[
                    ("text_cleaning", text_cleaner),
                    ("vectorizer", CountVectorizer()),
                ]
            )
            return processor

        except Exception as e:
            raise SentimentAnalysisException(e, sys)
    

        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of Datatransformation class" )
        try:
            logging.info("Starting data transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)


            ##training dataframe
            # input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis =1)
            # target_feature_train_df = train_df[TARGET_COLUMN]
            # target_feature_train_df = target_feature_train_df.replace(-1,0)

            # input_feature_test_df=test_df.drop(columns=[TARGET_COLUMN], axis =1)
            # target_feature_test_df = test_df[TARGET_COLUMN]
            # target_feature_test_df = target_feature_test_df.replace(-1,0)
            train_df[TARGET_COLUMN] = train_df["rating"].apply(lambda x: 1 if x > 3 else 0).astype(int)
            test_df[TARGET_COLUMN] = test_df["rating"].apply(lambda x: 1 if x > 3 else 0).astype(int)

            # ✅ Split X/y using TARGET_COLUMN (so downstream keeps working)
            X_train = train_df[["reviewText"]]
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df[["reviewText"]]
            y_test = test_df[TARGET_COLUMN]

            preprocessor = self.get_data_transformer_object()

            # preprocessor_object = preprocessor.fit(input_feature_train_df)
            # transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            # transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            # test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]
                        # Fit on train, transform train & test
            X_train_t = preprocessor.fit_transform(X_train).toarray()
            X_test_t = preprocessor.transform(X_test).toarray()

            # Concatenate X and y
            train_arr = np.c_[X_train_t, np.array(y_train)]
            test_arr = np.c_[X_test_t, np.array(y_test)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr,)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr,)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            save_object( "final_model/preprocessor.pkl",preprocessor )

            "Preparing Artifact"

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact


        except Exception as e:
            raise SentimentAnalysisException(e,sys)