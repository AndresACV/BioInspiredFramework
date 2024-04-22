# Proyecto-Academico\src\ModelEvaluationComponent\pytorch_model.py

import torch
import torch.nn as nn

class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)