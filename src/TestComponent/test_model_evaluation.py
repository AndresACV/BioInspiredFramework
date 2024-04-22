import sys
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

# Añade el directorio raíz del proyecto a sys.path
project_root = Path(__file__).resolve().parents[1] 
sys.path.append(str(project_root))

# Importaciones necesarias de sklearn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importaciones de los componentes del proyecto
from ModelEvaluationComponent.model_evaluator import ModelEvaluator
from ModelEvaluationComponent.pytorch_model import SimpleLinearModel
from ModelEvaluationComponent.tensorflow_model import create_simple_linear_model
import torch
import torch.optim as optim
import numpy as np

# Paso 2: Preparar Datos de Prueba
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Paso 3: Entrenamiento de los Modelos

# Entrenamiento en PyTorch
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

# Entrenamiento en TensorFlow
tensorflow_model = create_simple_linear_model(input_dim=1, output_dim=1)
tensorflow_model.compile(optimizer='sgd', loss='mean_squared_error')
tensorflow_model.fit(X_train_scaled, y_train, epochs=100, verbose=0)

# Paso 4: Evaluar los Modelos

# Evaluar el modelo de PyTorch
pytorch_evaluator = ModelEvaluator(pytorch_model, framework='pytorch')
mse_pytorch, r2_pytorch = pytorch_evaluator.evaluate(X_test_scaled, np.array(y_test).reshape(-1, 1))
print(f"PyTorch Model - MSE: {mse_pytorch}, R2: {r2_pytorch}")

# Evaluar el modelo de TensorFlow
tensorflow_evaluator = ModelEvaluator(tensorflow_model, framework='tensorflow')
mse_tensorflow, r2_tensorflow = tensorflow_evaluator.evaluate(X_test_scaled, y_test)
print(f"TensorFlow Model - MSE: {mse_tensorflow}, R2: {r2_tensorflow}")

"""
El modelo de PyTorch tiene un error cuadrático medio (MSE) de aproximadamente 5.04 y un coeficiente de determinación (R2) de aproximadamente 0.983, 
lo que indica un buen ajuste. 

El modelo de TensorFlow, por otro lado, tiene un MSE mucho menor (aproximadamente 0.0067) y un R2 muy cercano a 1, 
lo que indica un ajuste excepcionalmente bueno a los datos de prueba.
"""