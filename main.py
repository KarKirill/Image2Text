import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Преобразования для данных
transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразует изображение в тензор и нормализует значения в [0, 1]
    transforms.Normalize((0.1307,), (0.3081,))  # Нормализация по mean и std MNIST
])

# Загрузка данных
train_dataset = datasets.MNIST(root='mnist', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='mnist', train=False, download=True, transform=transform)



# Создание DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Базовая модель
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Исправлено здесь
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = nn.functional.relu(self.bn3(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
    
        return x
    
    # def visualize_activations(self, dataset, num_images=5):
    #     """
    #     Визуализация исходных изображений, активаций после conv1 и conv2.
        
    #     Args:
    #         dataset: Объект Dataset (например, train_dataset).
    #         num_images: Количество изображений для отображения.
    #     """
    #     # Устанавливаем модель в режим оценки
    #     self.eval()

    #     # Создаем фигуру для визуализации
    #     fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    #     fig.tight_layout()

    #     for i in range(min(num_images, len(dataset))):
    #         # Получаем изображение и метку из датасета
    #         x, y = dataset[i]
    #         original_image = x.squeeze().cpu().numpy()  # Исходное изображение (H, W)
    #         x = x.unsqueeze(0)  # Добавляем размерность batch (1, C, H, W)

    #         # Пропускаем через conv1
    #         with torch.no_grad():
    #             activations_conv1 = self.conv1(x).squeeze(0).cpu().numpy()  # (C, H, W)

    #         # Пропускаем через conv2
    #         with torch.no_grad():
    #             activations_conv2 = self.conv2(self.conv1(x)).squeeze(0).cpu().numpy()  # (C, H, W)

    #         # Отображаем исходное изображение
    #         axes[i, 0].imshow(original_image, cmap='gray')
    #         axes[i, 0].set_title(f'Original Image\nLabel: {y}')
    #         axes[i, 0].axis('off')

    #         # Отображаем активации после conv1
    #         activations_to_show = min(activations_conv1.shape[0], 10)  # Максимум 10 каналов
    #         for j in range(activations_to_show):
    #             axes[i, 1].imshow(activations_conv1[j], cmap='viridis')
    #             axes[i, 1].set_title(f'Conv1 Activations\nChannel {j + 1}')
    #             axes[i, 1].axis('off')
    #             break  # Показываем только первый канал для наглядности

    #         # Отображаем активации после conv2
    #         activations_to_show = min(activations_conv2.shape[0], 10)  # Максимум 10 каналов
    #         for j in range(activations_to_show):
    #             axes[i, 2].imshow(activations_conv2[j], cmap='viridis')
    #             axes[i, 2].set_title(f'Conv2 Activations\nChannel {j + 1}')
    #             axes[i, 2].axis('off')
    #             break  # Показываем только первый канал для наглядности

    #     plt.show()

# Модель с увеличенным количеством фильтров
class LargerFiltersModel(nn.Module):
    def __init__(self):
        super(LargerFiltersModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Исправлено здесь
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = x.view(-1, 128 * 7 * 7)  # Flatten
        x = nn.functional.relu(self.bn3(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
# Модель с дополнительным полносвязным слоем
class DeeperFCModel(nn.Module):
    def __init__(self):
        super(DeeperFCModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Исправлено здесь
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = nn.functional.relu(self.bn3(self.fc1(x)))
        x = self.dropout2(x)
        x = nn.functional.relu(self.bn4(self.fc2(x)))
        x = self.dropout3(x)
        x = self.fc3(x)
        return x
    
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()  # Переводим модель в режим обучения
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Обучение на тренировочных данных
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Обнуляем градиенты
            outputs = model(inputs)  # Прямой проход
            loss = criterion(outputs, labels)  # Вычисление потерь
            loss.backward()  # Обратный проход
            optimizer.step()  # Обновление весов
            
            # Сбор статистики
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Вычисление потерь и точности на тренировочных данных
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Валидация на тестовых данных
        model.eval()  # Переводим модель в режим оценки
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # Отключаем вычисление градиентов
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Вычисление потерь и точности на валидационных данных
        val_loss /= len(test_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Вывод статистики
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Определение функции потерь
criterion = nn.CrossEntropyLoss()

# Создание моделей
base_model = BaseModel()
larger_filters_model = LargerFiltersModel()
deeper_fc_model = DeeperFCModel()

# Обучение и оценка моделей
models = {
    "BaseModel": base_model,
    "LargerFiltersModel": larger_filters_model,
    "DeeperFCModel": deeper_fc_model
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses, val_losses, train_accuracies, val_accuracies = train_and_evaluate(
        model, train_loader, test_loader, criterion, optimizer, num_epochs=1
    )
    results[name] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies
    }

# Построение графиков для сравнения моделей
plt.figure(figsize=(15, 10))

for i, (name, result) in enumerate(results.items()):
    plt.subplot(2, 2, i+1)
    plt.plot(result['train_losses'], label='Train Loss')
    plt.plot(result['val_losses'], label='Val Loss')
    plt.title(f'{name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, i+2)
    plt.plot(result['train_accuracies'], label='Train Accuracy')
    plt.plot(result['val_accuracies'], label='Val Accuracy')
    plt.title(f'{name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

plt.tight_layout()
plt.show()


# Функция для вычисления метрик
def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    return accuracy, precision, recall, f1

# Оценка моделей
final_results = {}

for name, model in models.items():
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader)
    final_results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

# Вывод результатов
for name, metrics in final_results.items():
    print(f"{name}:")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1-Score: {metrics['F1-Score']:.4f}")
    
# model.visualize_activations(train_dataset, num_images=5)