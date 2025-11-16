import torch
import torch.optim as optim
from .models import create_model
from .utils import calculate_accuracy, load_data
from .config import config

class HyperparameterTuner:
    def __init__(self):
        self.results = []
    
    def grid_search(self, dataset_name='fashion_mnist'):
        """Запуск Grid Search"""
        best_accuracy = 0
        best_params = {}
        
        for lr in config.LEARNING_RATES:
            for batch_size in config.BATCH_SIZES:
                for optimizer_name in config.OPTIMIZERS:
                    print(f"\n🔍 Testing: LR={lr}, BS={batch_size}, Optimizer={optimizer_name}")
                    
                    # Загрузка данных
                    train_loader, test_loader, _ = load_data(dataset_name, batch_size)
                    
                    # Создание модели
                    model = create_model('medium').to(config.DEVICE)
                    
                    # Оптимизатор
                    if optimizer_name == 'SGD':
                        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                    else:
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                    
                    criterion = torch.nn.CrossEntropyLoss()
                    
                    # Обучение
                    accuracy = self._train_model(model, train_loader, test_loader, 
                                               optimizer, criterion, epochs=5)
                    
                    # Сохранение результатов
                    result = {
                        'lr': lr,
                        'batch_size': batch_size,
                        'optimizer': optimizer_name,
                        'accuracy': accuracy
                    }
                    self.results.append(result)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = result
        
        return best_params, self.results
    
    def _train_model(self, model, train_loader, test_loader, optimizer, criterion, epochs=5):
        """Обучение одной модели"""
        model.train()
        
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return calculate_accuracy(model, test_loader, config.DEVICE)