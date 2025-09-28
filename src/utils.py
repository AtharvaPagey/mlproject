import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomExeception
from sklearn.metrics import r2_score
import dill
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomExeception(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(list(models.values()))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            cv = GridSearchCV(estimator=model, param_grid=param, n_jobs=-1, cv = 3, verbose= 3, refit=True)
            cv.fit(X_train, y_train)

            model.set_params(**cv.best_params_)
            model.fit(X_train, y_train)
            
            y_test_pred = model.predict(X_test)

            test_score = r2_score(y_test_pred, y_test)

            report[list(models.keys())[i]] = (test_score, cv.best_params_)
        
        return report

    except Exception as e:
        raise CustomExeception(e, sys)