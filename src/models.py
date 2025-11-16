import torch.nn as nn
from .config import config

class AdvancedNet(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128, 64], 
                 output_size=10, dropout_rate=0.5, activation='relu'):
        super(AdvancedNet, self).__init__()
        
        # Выбор функции активации
        activation_functions = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
        }
        self.activation = activation_functions.get(activation, nn.ReLU())
        
        # Динамическое создание слоев
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Выходной слой
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

# Фабрика моделей для экспериментов
def create_model(model_type='medium', activation='relu'):
    architectures = {
        'small': [128, 64],
        'medium': [256, 128, 64], 
        'large': [512, 256, 128, 64]
    }
    
    return AdvancedNet(
        hidden_sizes=architectures[model_type],
        dropout_rate=0.5,
        activation=activation
    )