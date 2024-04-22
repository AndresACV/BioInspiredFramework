# Proyecto-Academico\src\BenchmarkingComponent\hyperparameter_optimizer.py

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class HyperparameterOptimizer:
    def __init__(self, estimator, param_grid, cv=5, search_type='grid'):
        """
        Initializes the HyperparameterOptimizer with an estimator, parameter grid, and cross-validation strategy.
        :param estimator: The machine learning estimator.
        :param param_grid: The parameter grid to search over.
        :param cv: The number of cross-validation folds.
        :param search_type: The type of search ('grid' or 'random').
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.search_type = search_type

    def optimize(self, X, y):
        """
        Performs hyperparameter optimization and returns the best estimator.
        :param X: Training features.
        :param y: Training target.
        :return: The best estimator after optimization.
        """
        if self.search_type == 'grid':
            search = GridSearchCV(self.estimator, self.param_grid, cv=self.cv)
        elif self.search_type == 'random':
            search = RandomizedSearchCV(self.estimator, self.param_grid, cv=self.cv, n_iter=10)
        else:
            raise ValueError("Unsupported search type. Choose 'grid' or 'random'.")

        search.fit(X, y)
        return search.best_estimator_