import sys, os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        cp: int,
        restecg: int,
        slp: int,
        caa: int,
        thall: int,
        fbs: int,
        sex: int,
        exng: int,
        age: int,
        trtbps: int,
        chol: int,
        thalachh: int,
        oldpeak: float,
    ):
        self.cp = cp

        self.restecg = restecg

        self.slp = slp

        self.caa = caa

        self.thall = thall

        self.fbs = fbs

        self.sex = sex

        self.exng = exng

        self.age = age

        self.trtbps = trtbps

        self.chol = chol

        self.thalachh = thalachh

        self.oldpeak = oldpeak

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "cp": [self.cp],
                "restecg": [self.restecg],
                "slp": [self.slp],
                "caa": [self.caa],
                "thall": [self.thall],
                "fbs": [self.fbs],
                "sex": [self.sex],
                "exng": [self.exng],
                "age": [self.age],
                "trtbps": [self.trtbps],
                "chol": [self.chol],
                "thalachh": [self.thalachh],
                "oldpeak": [self.oldpeak],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

