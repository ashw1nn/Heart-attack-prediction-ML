import sys, os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path =  os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "age",
                "trtbps",
                "chol",
                "thalachh",
                "oldpeak",
            ]
            categorical_columns = [
                "cp",
                "restecg",
                "slp",
                "caa",
                "thall",
                "fbs",
                "sex",
                "exng",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", MinMaxScaler()),
                ]
            )


            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder()),
                ]
            )

            logging.info(f"Numerical coloumns: {numerical_columns}")
            logging.info(f"Categorical coloumns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read test and train data completed")
            logging.info("Obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformer_object()

            target_coloumn_name = "output"


            input_feature_train_df = train_df.drop(target_coloumn_name, axis=1)
            target_feature_train_df = train_df[target_coloumn_name]

            input_feature_test_df = test_df.drop(target_coloumn_name, axis=1)
            target_feature_test_df = test_df[target_coloumn_name]

            logging.info("Applying preprocessor on train and test data")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

