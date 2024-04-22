import sys
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

# Add the project root folder to the path in order to be able to import the required modules
project_root = Path(__file__).resolve().parents[1] 
sys.path.append(str(project_root))

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ModelEvaluationComponent.model_evaluator import ModelEvaluator
from ModelEvaluationComponent.pytorch_model import SimpleLinearModel
from ModelEvaluationComponent.tensorflow_model import create_simple_linear_model
import torch
import torch.optim as optim
import numpy as np

# Data Preparation
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the Models

# Training in PyTorch
pytorch_model = SimpleLinearModel(input_dim=1, output_dim=1)
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(pytorch_model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = pytorch_model(torch.from_numpy(X_train_scaled).float())
    loss = criterion(outputs, torch.from_numpy(y_train).float().view(-1, 1))
    loss.backward()
    optimizer.step()

# Training in TensorFlow
tensorflow_model = create_simple_linear_model(input_dim=1, output_dim=1)
tensorflow_model.compile(optimizer='sgd', loss='mean_squared_error')
tensorflow_model.fit(X_train_scaled, y_train, epochs=100, verbose=0)

# Paso 4: Evaluation of the models

# Evaluation of the PyTorch model
pytorch_evaluator = ModelEvaluator(pytorch_model, framework='pytorch')
mse_pytorch, r2_pytorch = pytorch_evaluator.evaluate(X_test_scaled, np.array(y_test).reshape(-1, 1))
print(f"PyTorch Model - MSE: {mse_pytorch}, R2: {r2_pytorch}")

# Evaluation of the TensorFlow model
tensorflow_evaluator = ModelEvaluator(tensorflow_model, framework='tensorflow')
mse_tensorflow, r2_tensorflow = tensorflow_evaluator.evaluate(X_test_scaled, y_test)
print(f"TensorFlow Model - MSE: {mse_tensorflow}, R2: {r2_tensorflow}")

"""
The PyTorch model has a mean squared error (MSE) of approximately 5.04 and a coefficient of determination (R2) of approximately 0.983, indicating a good fit.

The TensorFlow model, on the other hand, has a much lower MSE (approximately 0.0067) and an R2 very close to 1, indicating an exceptionally good fit to the test data.
"""