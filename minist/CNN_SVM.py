import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return x


class SVM(nn.Module):
    def __init__(self, input_size=512, num_classes=10):
        super(SVM, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


def svm_loss(outputs, labels, C=1.0):
    batch_size = outputs.size(0)
    # 获取正确类别的分数
    correct_scores = outputs[torch.arange(batch_size), labels]
    # 计算边界损失
    margins = torch.clamp(outputs - correct_scores.view(-1, 1) + 1.0, min=0.0)
    # 将正确类别的margin设为0
    margins[torch.arange(batch_size), labels] = 0
    # 计算铰链损失
    loss = margins.sum() / batch_size
    # 添加正则化项
    loss += C * (outputs ** 2).sum() / (2.0 * batch_size)
    return loss


class CNNSVM:
    def __init__(self, save_dir='results_CNN_SVM'):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.feature_extractor = CNNFeatureExtractor().to(self.device)
        self.svm = SVM().to(self.device)
        self.is_trained = False

        print(f"Using device: {self.device}")

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.history = {
            'losses': [],
            'accuracies': []
        }

    def save_results(self, y_true, y_pred, y_scores):
        """保存所有评估结果和可视化"""

        # 1. 保存模型文件
        joblib.dump(self.feature_extractor, os.path.join(self.save_dir, 'adaboost_model.joblib'))
        joblib.dump(self.svm, os.path.join(self.save_dir, 'adaboost_scaler.joblib'))

        # 2. 绘制并保存混淆矩阵
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
        plt.close()

        # 3. 保存评估报告
        report = classification_report(y_true, y_pred)
        with open(os.path.join(self.save_dir, 'evaluation_report.txt'), 'w') as f:
            f.write(report)

        # 4. 绘制并保存学习曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['losses'], label='Training Loss')
        plt.plot(self.history['accuracies'], label='Accuracy')
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'learning_curve_simple.png'))
        plt.close()

        # 5. 保存样本预测结果可视化
        plt.figure(figsize=(12, 6))
        n_samples = min(10, len(y_true))
        indices = np.random.choice(len(y_true), n_samples, replace=False)

        plt.subplot(1, 2, 1)
        plt.bar(range(n_samples), y_true[indices], label='True')
        plt.bar(range(n_samples), y_pred[indices], label='Predicted', alpha=0.5)
        plt.title('Sample Predictions Comparison')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(y_scores[indices].T)
        plt.title('Prediction Probabilities')
        plt.savefig(os.path.join(self.save_dir, 'sample_predictions.png'))
        plt.close()

    def evaluate(self, test_loader):
        self.feature_extractor.eval()
        self.svm.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                features = self.feature_extractor(data)
                outputs = self.svm(features)
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = 100. * correct / total
        return accuracy

    def train(self, train_loader, test_loader, epochs=10):
        # 首先预训练特征提取器
        temp_classifier = nn.Linear(512, 10).to(self.device)
        criterion = nn.CrossEntropyLoss()
        feature_optimizer = optim.Adam(self.feature_extractor.parameters())
        classifier_optimizer = optim.Adam(temp_classifier.parameters())

        print("预训练特征提取器...")
        for epoch in range(5):
            self.feature_extractor.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                feature_optimizer.zero_grad()
                classifier_optimizer.zero_grad()

                features = self.feature_extractor(data)
                output = temp_classifier(features)

                loss = criterion(output, target)
                loss.backward()

                feature_optimizer.step()
                classifier_optimizer.step()

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                if (batch_idx + 1) % 100 == 0:
                    print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {loss.item():.4f}')

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            print(f'Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%')

        # 训练SVM分类器
        print("训练SVM分类器...")
        svm_optimizer = optim.Adam(self.svm.parameters())

        for epoch in range(epochs):
            self.feature_extractor.eval()
            self.svm.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                with torch.no_grad():
                    features = self.feature_extractor(data)

                svm_optimizer.zero_grad()
                outputs = self.svm(features)
                loss = svm_loss(outputs, target)
                loss.backward()
                svm_optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                if (batch_idx + 1) % 100 == 0:
                    print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {loss.item():.4f}')

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total

            self.history['losses'].append(epoch_loss)
            self.history['accuracies'].append(epoch_acc)

            test_acc = self.evaluate(test_loader)
            print(
                f'Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Train Acc = {epoch_acc:.2f}%, Test Acc = {test_acc:.2f}%')

        # 最终评估
        self.feature_extractor.eval()
        self.svm.eval()
        y_true = []
        y_pred = []
        y_scores = []

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                features = self.feature_extractor(data)
                outputs = self.svm(features)
                scores = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_scores.extend(scores.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)

        self.save_results(y_true, y_pred, y_scores)

        self.is_trained = True
        print("模型训练完成并保存所有结果！")


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # 创建保存目录
    save_dir = os.path.join('results_CNN_SVM', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)

    # 训练模型
    model = CNNSVM(save_dir=save_dir)
    model.train(train_loader, test_loader, epochs=10)

    print(f"所有结果已保存到: {save_dir}")


if __name__ == "__main__":
    main()