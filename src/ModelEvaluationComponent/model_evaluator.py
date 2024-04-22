# Proyecto-Academico\src\ModelEvaluationComponent\model_evaluator.py

import torch
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

class ModelEvaluator:
    def __init__(self, model, framework='pytorch'):
        """
        Initializes the ModelEvaluator with a model and the framework.
        :param model: The model to be evaluated.
        :param framework: The framework used ('pytorch' or 'tensorflow').
        """
        self.model = model
        self.framework = framework

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on the test set and returns the MSE and R2 score.
        :param X_test: Test features.
        :param y_test: True values for the test set.
        :return: A tuple containing the MSE and R2 score.
        """
        if self.framework == 'pytorch':
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                predictions = self.model(torch.from_numpy(X_test).float()).numpy()
        elif self.framework == 'tensorflow':
            predictions = self.model.predict(X_test)
        else:
            raise ValueError("Unsupported framework. Choose 'pytorch' or 'tensorflow'.")

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        return mse, r2