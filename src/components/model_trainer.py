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
            }

            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
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
        
