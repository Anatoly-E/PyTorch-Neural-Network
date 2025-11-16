import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import json
import argparse
from .models import create_model, AdvancedNet
from .utils import load_data, calculate_accuracy, EarlyStopping, visualize_dataset
from .hyperparameter_tuning import HyperparameterTuner
from .config import config

def parse_arguments():
    parser = argparse.ArgumentParser(description='FashionMNIST Experiments')
    parser.add_argument('--experiments', type=str, nargs='+', 
                       choices=['1', '2', '3', '4', '5', 'all'],
                       default=['all'],
                       help='Which experiments to run (1-5 or all)')
    return parser.parse_args()

class FashionMNISTTrainer:
    def __init__(self):
        self.device = config.DEVICE
        os.makedirs('./outputs/models', exist_ok=True)
        os.makedirs('./outputs/plots', exist_ok=True)
        os.makedirs('./outputs/results', exist_ok=True)
    
    def experiment_1_dataset_comparison(self):
        """Эксперимент 1: Сравнение MNIST и FashionMNIST"""
        print("=" * 60)
        print("ЭКСПЕРИМЕНТ 1: Сравнение датасетов")
        print("=" * 60)
        
        # Загрузка обоих датасетов
        fashion_train, fashion_test, fashion_dataset = load_data('fashion_mnist', 64)
        mnist_train, mnist_test, mnist_dataset = load_data('mnist', 64)
        
        # Визуализация
        visualize_dataset(fashion_dataset, 'FashionMNIST')
        visualize_dataset(mnist_dataset, 'MNIST')
        
        print("\n📊 Обучение на FashionMNIST...")
        # Модель для FashionMNIST
        model_fashion = create_model('medium').to(self.device)
        optimizer_fashion = optim.Adam(model_fashion.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        fashion_acc = self._quick_train(model_fashion, fashion_train, fashion_test, 
                                    optimizer_fashion, criterion)
        
        print("\n📊 Обучение на MNIST...")
        # ОТДЕЛЬНАЯ модель для MNIST
        model_mnist = create_model('medium').to(self.device)
        optimizer_mnist = optim.Adam(model_mnist.parameters(), lr=0.001)
        
        mnist_acc = self._quick_train(model_mnist, mnist_train, mnist_test, 
                                    optimizer_mnist, criterion)
        
        print(f"\n📈 РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
        print(f"FashionMNIST точность: {fashion_acc:.2f}%")
        print(f"MNIST точность: {mnist_acc:.2f}%")
        print(f"Разница: {abs(fashion_acc - mnist_acc):.2f}%")
        
        # Дополнительно: тестирование перекрестно (для демонстрации)
        print(f"\n🔍 Перекрестное тестирование (для демонстрации проблемы):")
        print(f"Модель FashionMNIST на данных MNIST: "
            f"{calculate_accuracy(model_fashion, mnist_test, self.device):.2f}%")
        print(f"Модель MNIST на данных FashionMNIST: "
            f"{calculate_accuracy(model_mnist, fashion_test, self.device):.2f}%")
        
    def experiment_2_architectures(self):
        """Эксперимент 2: Разные архитектуры"""
        print("\n" + "=" * 60)
        print("ЭКСПЕРИМЕНТ 2: Разные архитектуры моделей")
        print("=" * 60)
        
        train_loader, test_loader, _ = load_data('fashion_mnist', 64)
        
        architectures = {
            # 1. Разные размеры архитектур
            'small': create_model('small'),
            'medium': create_model('medium'), 
            'large': create_model('large'),
            
            # 2. Разные функции активации (на одной архитектуре)
            'relu': create_model('medium', 'relu'),
            'gelu': create_model('medium', 'gelu'),
            'tanh': create_model('medium', 'tanh'),        }
        
        results = {}
        for name, model in architectures.items():
            print(f"\n🏗️ Тестирование архитектуры: {name}")
            model = model.to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            accuracy = self._quick_train(model, train_loader, test_loader, optimizer, criterion)
            results[name] = accuracy
            print(f"Точность {name}: {accuracy:.2f}%")
        
        # Сохранение результатов
        with open('./outputs/results/architecture_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Визуализация
        self._plot_architecture_results(results)

    def experiment_3_hyperparameter_tuning(self):
        """Эксперимент 3: Настройка гиперпараметров"""
        print("\n" + "=" * 60)
        print("ЭКСПЕРИМЕНТ 3: Grid Search гиперпараметров")
        print("=" * 60)
        
        tuner = HyperparameterTuner()
        best_params, all_results = tuner.grid_search('fashion_mnist')
        
        print(f"\n🎯 ЛУЧШИЕ ПАРАМЕТРЫ:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # Сохранение результатов
        with open('./outputs/results/grid_search_results.json', 'w') as f:
            json.dump({
                'best_params': best_params,
                'all_results': all_results
            }, f, indent=2)
    
    def experiment_4_early_stopping(self):
        """Эксперимент 4: Ранняя остановка"""
        print("\n" + "=" * 60)
        print("ЭКСПЕРИМЕНТ 4: Early Stopping")
        print("=" * 60)
        
        train_loader, test_loader, _ = load_data('fashion_mnist', 64)
        model = create_model('medium').to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        
        early_stopping = EarlyStopping(patience=2, min_delta=0.01)
        train_losses, val_accuracies = [], []
        
        for epoch in range(20):  # Максимум 20 эпох
            # Обучение
            model.train()
            epoch_loss = 0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Валидация
            accuracy = calculate_accuracy(model, test_loader, self.device)
            train_losses.append(epoch_loss / len(train_loader))
            val_accuracies.append(accuracy)
            
            print(f'Epoch {epoch+1}: Loss: {train_losses[-1]:.4f}, Acc: {accuracy:.2f}%')
            
            # Проверка ранней остановки
            if early_stopping(train_losses[-1], model):
                print(f'🛑 Early stopping на эпохе {epoch+1}!')
                break
        
        self._plot_training_curves(train_losses, val_accuracies)
        
    def experiment_5_lr_scheduling(self):
        """Эксперимент 5: Планировщик Learning Rate с TensorBoard"""
        print("\n" + "=" * 60)
        print("ЭКСПЕРИМЕНТ 5: LR Scheduling with TensorBoard")
        print("=" * 60)
        
        # 🔥 ДОБАВЛЕНО: Инициализация TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"./outputs/tensorboard/lr_scheduling_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"📊 TensorBoard: {log_dir}")
        
        # Ваш существующий код без изменений...
        train_loader, test_loader, _ = load_data('fashion_mnist', 64)
        
        schedulers_config = [
            {'name': 'StepLR', 'scheduler': None, 'kwargs': {'step_size': 3, 'gamma': 0.1}},
            {'name': 'ReduceLROnPlateau', 'scheduler': None, 'kwargs': {'mode': 'min', 'patience': 2, 'factor': 0.5}}
        ]
        
        for config in schedulers_config:
            print(f"\n📉 Тестирование {config['name']}...")
            
            model = create_model('medium').to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()
            
            if config['name'] == 'StepLR':
                scheduler = StepLR(optimizer, **config['kwargs'])
            else:
                scheduler = ReduceLROnPlateau(optimizer, **config['kwargs'])
            
            learning_rates = []
            train_losses = []
            accuracies = []
            
            for epoch in range(8):
                model.train()
                epoch_loss = 0
                for data, target in train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_epoch_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_epoch_loss)
                
                accuracy = calculate_accuracy(model, test_loader, self.device)
                accuracies.append(accuracy)
                
                if config['name'] == 'StepLR':
                    scheduler.step()
                else:
                    scheduler.step(avg_epoch_loss)
                
                current_lr = optimizer.param_groups[0]['lr']
                learning_rates.append(current_lr)
                
                # 🔥 ДОБАВЛЕНО: Логирование в TensorBoard
                writer.add_scalar(f'LR/{config["name"]}', current_lr, epoch)
                writer.add_scalar(f'Loss/{config["name"]}', avg_epoch_loss, epoch)
                writer.add_scalar(f'Accuracy/{config["name"]}', accuracy, epoch)
                
                print(f'Epoch {epoch+1}: LR={current_lr:.6f}, Loss: {avg_epoch_loss:.4f}, Acc: {accuracy:.2f}%')
            
            self._plot_lr_schedule(learning_rates, accuracies, config['name'])
            self._plot_lr_training_curves(learning_rates, train_losses, accuracies, config['name'])
        
        # 🔥 ДОБАВЛЕНО: Закрытие writer
        writer.close()
        print(f"✅ TensorBoard логи сохранены. Запустите: tensorboard --logdir=./outputs/tensorboard")

    def _plot_lr_training_curves(self, learning_rates, losses, accuracies, scheduler_name):
        """Визуализация кривых обучения с LR"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        
        # Learning Rate
        ax1.semilogy(learning_rates, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Learning Rate')
        ax1.set_title(f'LR Schedule: {scheduler_name}')
        ax1.grid(True)
        
        # Loss
        ax2.plot(losses, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Training Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True)
        
        # Accuracy
        ax3.plot(accuracies, 'g-o', linewidth=2, markersize=6)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Validation Accuracy')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'./outputs/plots/lr_training_curves_{scheduler_name}.png')
        plt.show()  

    def _quick_train(self, model, train_loader, test_loader, optimizer, criterion, epochs=3):
        """Быстрое обучение для экспериментов"""
        model.train()
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return calculate_accuracy(model, test_loader, self.device)
    
    def _plot_architecture_results(self, results):
        """Визуализация результатов архитектур"""
        plt.figure(figsize=(10, 6))
        names = list(results.keys())
        accuracies = list(results.values())
        
        bars = plt.bar(names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'violet'])
        plt.ylabel('Точность (%)')
        plt.title('Сравнение архитектур моделей на FashionMNIST')
        plt.ylim(0, 100)
        
        # Добавление значений на столбцы
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('./outputs/plots/architecture_comparison.png')
        plt.show()
    
    def _plot_training_curves(self, losses, accuracies):
        """Визуализация кривых обучения"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(losses, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True)
        
        ax2.plot(accuracies, 'r-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('./outputs/plots/training_curves.png')
        plt.show()
    
    def _plot_lr_schedule(self, learning_rates, accuracies, scheduler_name):
        """Визуализация изменения LR"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.semilogy(learning_rates, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Learning Rate')
        ax1.set_title(f'LR Schedule: {scheduler_name}')
        ax1.grid(True)
        
        ax2.plot(accuracies, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Model Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'./outputs/plots/lr_schedule_{scheduler_name}.png')
        plt.show()

def main():
    """Главная функция для запуска всех экспериментов"""
    args = parse_arguments()
    trainer = FashionMNISTTrainer()
    
    experiments_to_run = {
        '1': trainer.experiment_1_dataset_comparison,
        '2': trainer.experiment_2_architectures,
        '3': trainer.experiment_3_hyperparameter_tuning, 
        '4': trainer.experiment_4_early_stopping,
        '5': trainer.experiment_5_lr_scheduling
    }
    
    if 'all' in args.experiments:
        # Запуск всех экспериментов
        for exp_num, exp_func in experiments_to_run.items():
            print(f"\n🚀 Запуск эксперимента {exp_num}...")
            exp_func()
    else:
        # Запуск только выбранных экспериментов
        for exp_num in args.experiments:
            if exp_num in experiments_to_run:
                print(f"\n🚀 Запуск эксперимента {exp_num}...")
                experiments_to_run[exp_num]()
    
    print("\n🎉 Выбранные эксперименты завершены!")

if __name__ == "__main__":
    main()