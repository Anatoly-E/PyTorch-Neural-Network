import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# -------------------------------
# Конфигурация
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]
num_epochs = 5

# Папка для сохранения графиков
os.makedirs("../results", exist_ok=True)

# -------------------------------
# Подготовка данных
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="../data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="../data", train=False, transform=transform, download=True)

# -------------------------------
# Определение архитектуры сети
# -------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------------------
# Функции обучения и тестирования
# -------------------------------
def train_and_evaluate(lr, batch_size):
    print(f"\n[INFO] Training with lr={lr}, batch_size={batch_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        test_accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Сохраняем графики
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.title(f"Loss (lr={lr}, bs={batch_size})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.title(f"Accuracy (lr={lr}, bs={batch_size})")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"../results/lr_{lr}_bs_{batch_size}.png")
    plt.close()

    # Возвращаем итоговую точность
    return test_accuracies[-1]

# -------------------------------
# Основной цикл экспериментов
# -------------------------------
results = []

for lr in learning_rates:
    for bs in batch_sizes:
        acc = train_and_evaluate(lr, bs)
        results.append((lr, bs, acc))

# -------------------------------
# Вывод итогов
# -------------------------------
print("\n===== Итоговая таблица точности =====")
print("LearningRate | BatchSize | Accuracy (%)")
for lr, bs, acc in results:
    print(f"{lr:<13} {bs:<10} {acc:.2f}")
