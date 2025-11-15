import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# Шаг 1: Загрузка датасета MNIST
def load_mnist(batch_size):
    """Загрузка и подготовка датасета MNIST"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

# Шаг 2: Построение полносвязной сети
class ThreeLayerNet(nn.Module):
    def __init__(self, input_size=784, hidden1=128, hidden2=64, output_size=10):
        super(ThreeLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten изображение 28x28 в вектор 784
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Функция для вычисления точности
def calculate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

# Функция обучения модели
def train_model(learning_rate, batch_size, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    print(f"Параметры: LR={learning_rate}, Batch Size={batch_size}")
    
    # Загрузка данных
    train_loader, test_loader = load_mnist(batch_size)
    
    # Создание модели
    model = ThreeLayerNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Для записи метрик
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward pass и оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # Вычисление метрик для эпохи
        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        epoch_test_acc = calculate_accuracy(model, test_loader, device)
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_acc)
        test_accuracies.append(epoch_test_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.2f}%, Test Acc: {epoch_test_acc:.2f}%')
    
    training_time = time.time() - start_time
    print(f"Обучение завершено за {training_time:.2f} секунд")
    print(f"Финальная точность на тесте: {test_accuracies[-1]:.2f}%")
    
    return train_losses, train_accuracies, test_accuracies, model

# Основной эксперимент
def run_experiments():
    # Шаг 3: Настройка гиперпараметров
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]
    num_epochs = 5
    
    results = {}
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"\n{'='*50}")
            print(f"ЭКСПЕРИМЕНТ: LR={lr}, Batch Size={batch_size}")
            print(f"{'='*50}")
            
            key = f"LR_{lr}_BS_{batch_size}"
            train_losses, train_acc, test_acc, model = train_model(
                learning_rate=lr, 
                batch_size=batch_size, 
                num_epochs=num_epochs
            )
            
            results[key] = {
                'train_losses': train_losses,
                'train_accuracies': train_acc,
                'test_accuracies': test_acc,
                'final_test_accuracy': test_acc[-1],
                'model': model
            }
    
    return results

# Визуализация результатов
def plot_results(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # График потерь для разных learning rates (при фиксированном batch size=64)
    ax1.set_title('Loss для разных Learning Rates (BS=64)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    for lr in [0.001, 0.01, 0.1]:
        key = f"LR_{lr}_BS_64"
        if key in results:
            ax1.plot(results[key]['train_losses'], label=f'LR={lr}')
    ax1.legend()
    ax1.grid(True)
    
    # График точности для разных learning rates
    ax2.set_title('Test Accuracy для разных Learning Rates (BS=64)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    for lr in [0.001, 0.01, 0.1]:
        key = f"LR_{lr}_BS_64"
        if key in results:
            ax2.plot(results[key]['test_accuracies'], label=f'LR={lr}')
    ax2.legend()
    ax2.grid(True)
    
    # График точности для разных batch sizes (при фиксированном lr=0.001)
    ax3.set_title('Test Accuracy для разных Batch Sizes (LR=0.001)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    for bs in [32, 64, 128]:
        key = f"LR_0.001_BS_{bs}"
        if key in results:
            ax3.plot(results[key]['test_accuracies'], label=f'BS={bs}')
    ax3.legend()
    ax3.grid(True)
    
    # Сводная таблица результатов
    final_accuracies = []
    configs = []
    for key, result in results.items():
        final_accuracies.append(result['final_test_accuracy'])
        configs.append(key)
    
    ax4.axis('off')
    table_data = []
    for config, acc in zip(configs, final_accuracies):
        table_data.append([config, f"{acc:.2f}%"])
    
    table = ax4.table(cellText=table_data, 
                     colLabels=['Configuration', 'Final Test Accuracy'], 
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.show()

# Запуск всей лабораторной работы
if __name__ == "__main__":
    print("Начало лабораторной работы: Исследование гиперпараметров для MNIST")
    results = run_experiments()
    plot_results(results)
    
    # Вывод лучшего результата
    best_config = max(results.items(), key=lambda x: x[1]['final_test_accuracy'])
    print(f"\n🎯 ЛУЧШИЙ РЕЗУЛЬТАТ:")
    print(f"Конфигурация: {best_config[0]}")
    print(f"Точность: {best_config[1]['final_test_accuracy']:.2f}%")