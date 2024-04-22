# Proyecto-Academico\src\test_hyperparameter_optimization.py

from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from BenchmarkingComponent.hyperparameter_optimizer import HyperparameterOptimizer

# Generar un conjunto de datos de prueba
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)

# Definir el modelo y la cuadrícula de parámetros
model = Ridge()
param_grid = {'alpha': [0.1, 1.0, 10.0]}

# Crear y ejecutar el optimizador
optimizer = HyperparameterOptimizer(model, param_grid, search_type='grid')
best_model = optimizer.optimize(X, y)

print(f"Mejor modelo: {best_model}")