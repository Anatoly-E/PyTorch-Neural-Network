import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def load_data(dataset_name='fashion_mnist', batch_size=64):
    """Загрузка датасетов"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)) if dataset_name == 'fashion_mnist' 
        else transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if dataset_name == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    else:  # mnist
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader, test_dataset

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model_state)
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
            
        return self.early_stop

def calculate_accuracy(model, data_loader, device):
    """Вычисление точности модели"""
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

def visualize_dataset(dataset, dataset_name='FashionMNIST'):
    """Визуализация датасета"""
    from .config import config
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        image, label = dataset[i]
        ax.imshow(image.squeeze(), cmap='gray')
        
        # ПРАВИЛЬНОЕ отображение названий классов
        if dataset_name == 'FashionMNIST':
            ax.set_title(f'{label}: {config.FASHION_CLASSES[label]}')
        else:
            ax.set_title(f'Digit: {label}')
        ax.axis('off')
    
    plt.suptitle(f'{dataset_name} Examples')
    plt.tight_layout()
    
    # Сохранение графика
    os.makedirs('./outputs/plots', exist_ok=True)
    plt.savefig(f'./outputs/plots/{dataset_name}_examples.png')
    plt.show()