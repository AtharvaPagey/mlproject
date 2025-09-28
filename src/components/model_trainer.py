import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomExeception
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Linear Regression":LinearRegression(),
                "Decision Tree Regressor" : DecisionTreeRegressor(),
                "Random Forest Regressor" : RandomForestRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor(),
                "CatBoost Regressor" : CatBoostRegressor(verbose = False),
                "KNeighbors Regressor" : KNeighborsRegressor(),
                "GradientBoostingRegressor" : GradientBoostingRegressor()
            }

            params = {
                "Linear Regression" : {
                    "fit_intercept" : [True, False],
                },
                "Decision Tree Regressor":{
                    "criterion" : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    "splitter" : ["best", "random"],
                    "max_depth" : [5, 6, 8, 7, 9, 10, 2, 3],
                    "max_features" : ["sqrt", "log2"]
                },
                "Random Forest Regressor" : {
                    "n_estimators" : [100, 150, 200, 175, 300, 325],
                    "criterion" : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    "max_depth" : [5, 6, 8, 7, 9, 10, 2, 3],
                    "max_features" : ["sqrt", "log2"]
                },
                "XGBRegressor" : {
                    # "loss" : ['squared_error', 'absolute_error', 'huber', 'quantile'],
                    "learning_rate" : [0.1, 0.01, 0.05, 0.15, 0.02],
                    "n_estimators" : [100, 150, 200, 175, 300, 325],
                    # "criterion" : ['squared_error', 'friedman_mse'],
                    "max_depth" : [5, 6, 8, 7, 9, 10, 2, 3],
                    # "max_features" : ["sqrt", "log2"]
                },
                "AdaBoost Regressor" : {
                    "loss" : ['linear', 'square', 'exponential'],
                    "learning_rate" : [0.1, 0.01, 0.05, 0.15, 0.02],
                    "n_estimators" : [100, 150, 200, 175, 300, 325]
                },
                "CatBoost Regressor" : {
                    "iterations" : [100, 150, 200, 175, 300, 325],
                    "learning_rate" : [0.1, 0.01, 0.05, 0.15, 0.02]
                },
                "KNeighbors Regressor" : {
                    "n_neighbors" : [5, 6, 7, 4, 3],
                    "weights" : ['uniform', 'distance'],
                    "algorithm" : ['auto'],
                    "p" : [1, 2]
                },
                "GradientBoostingRegressor" : {
                    "loss" : ['squared_error', 'absolute_error', 'huber', 'quantile'],
                    "learning_rate" : [0.1, 0.01, 0.05, 0.15, 0.02],
                    "n_estimators" : [100, 150, 200, 175, 300, 325]
                },
            }

            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models, params)

            best_model_name = max(model_report, key=lambda name: model_report[name][0])
            best_model_score = model_report[best_model_name][0]
            
            best_model = models[best_model_name]

            least_score = 0.6

            if best_model_score < least_score:
                raise CustomExeception("No Best Model Found")
            
            logging.info("Best model found o both training and testing")

            save_object(
                self.model_trainer_config.trained_model_file_path,
                best_model
            )

            predicted = best_model.predict(X_test)
            r2_val = r2_score(y_test, predicted)

            return r2_val

        except Exception as e:
            raise CustomExeception(e, sys)
        
