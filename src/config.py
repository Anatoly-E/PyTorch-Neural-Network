import torch

# Конфигурация проекта
class Config:
    # Пути
    DATA_PATH = './data'
    OUTPUT_PATH = './outputs'
    
    # Устройство
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Параметры данных
    FASHION_CLASSES = {
        0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 
        4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
    }
    
    # Гиперпараметры для экспериментов
    LEARNING_RATES = [0.1, 0.01, 0.001]
    BATCH_SIZES = [32, 64, 128]
    OPTIMIZERS = ['SGD', 'Adam']
    
    # Архитектуры моделей
    ARCHITECTURES = {
        'small': {'hidden_sizes': [128, 64], 'dropout': 0.3},
        'medium': {'hidden_sizes': [256, 128, 64], 'dropout': 0.5},
        'large': {'hidden_sizes': [512, 256, 128, 64], 'dropout': 0.5},
    }

config = Config()