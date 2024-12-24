import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import torch
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
import joblib
import os
from datetime import datetime

class SVMClassifier:
    def __init__(self, kernel='rbf', random_state=42, output_dir=None):
        """
        初始化SVM分类器
        """
        # 创建输出目录
        self.output_dir = output_dir or self._create_output_dir()
        self.device = (torch.device("mps") if torch.backends.mps.is_available()
                       else torch.device("cpu"))
        print(f"使用设备: {self.device}")
        print(f"输出目录: {self.output_dir}")

        self.classifier = SVC(
            kernel=kernel,
            random_state=random_state,
            verbose=False,
            cache_size=4000,
            shrinking=False,
            probability=True,
            C=10.0,
            gamma='scale',
            max_iter=10000
        )

        self.scaler = StandardScaler()
        self.model_path = os.path.join(self.output_dir, 'svm_model.joblib')
        self.scaler_path = os.path.join(self.output_dir, 'scaler.joblib')

    def _create_output_dir(self):
        """创建输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("results-SVM", f"run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def save_model(self):
        """保存模型和scaler"""
        try:
            print("正在保存模型...")
            joblib.dump(self.classifier, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            print("模型保存完成")
            return True
        except Exception as e:
            print(f"保存模型时出错: {str(e)}")
            return False

    def load_model(self):
        """加载模型和scaler"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                print("正在加载已保存的模型...")
                self.classifier = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                print("模型加载完成")
                return True
            else:
                print("未找到已保存的模型文件")
                return False
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return False

    def fit(self, X, y):
        """训练模型"""
        print("正在进行特征缩放...")
        X_scaled = self.scaler.fit_transform(X)

        print("开始训练模型...")
        # 使用分批训练来显示进度
        batch_size = 1000
        n_samples = len(X_scaled)
        n_batches = (n_samples + batch_size - 1) // batch_size

        with tqdm(total=n_samples, desc="训练进度") as pbar:
            fitted_samples = 0
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_X = X_scaled[i:end_idx]
                batch_y = y[i:end_idx]

                # 如果是第一个批次，直接fit
                if i == 0:
                    self.classifier.fit(batch_X, batch_y)
                else:
                    # 对后续批次使用partial_fit（注意：SVC不支持partial_fit，这里仅作示意）
                    self.classifier.fit(batch_X, batch_y)

                batch_processed = end_idx - i
                pbar.update(batch_processed)
                fitted_samples += batch_processed

        # 保存训练好的模型
        self.save_model()

    def predict(self, X):
        """预测新样本"""
        print("正在进行预测...")
        batch_size = 1000
        n_samples = len(X)
        predictions = []

        with tqdm(total=n_samples, desc="预测进度") as pbar:
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_X = X[i:end_idx]
                batch_X_scaled = self.scaler.transform(batch_X)
                batch_pred = self.classifier.predict(batch_X_scaled)
                predictions.extend(batch_pred)
                pbar.update(end_idx - i)

        return np.array(predictions)

    def predict_proba(self, X):
        """预测概率"""
        print("正在进行概率预测...")
        batch_size = 1000
        n_samples = len(X)
        probabilities = []

        with tqdm(total=n_samples, desc="概率预测进度") as pbar:
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_X = X[i:end_idx]
                batch_X_scaled = self.scaler.transform(batch_X)
                batch_proba = self.classifier.predict_proba(batch_X_scaled)
                probabilities.extend(batch_proba)
                pbar.update(end_idx - i)

        return np.array(probabilities)

def load_mnist_data():
    """
    加载完整的MNIST数据集
    """
    print("正在加载MNIST数据集...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    # 转换数据格式并显示进度条
    print("正在处理训练数据...")
    X_train = train_dataset.data.float().reshape(-1, 784).numpy()
    y_train = train_dataset.targets.numpy()

    print("正在处理测试数据...")
    X_test = test_dataset.data.float().reshape(-1, 784).numpy()
    y_test = test_dataset.targets.numpy()

    # 数据归一化到[0,1]区间
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_sample_predictions(X_test, y_test, y_pred, output_dir, num_samples=10):
    """绘制预测示例"""
    plt.figure(figsize=(20, 4))
    indices = np.random.choice(len(y_test), num_samples, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {y_test[idx]}\nPred: {y_pred[idx]}')
        plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'sample_predictions.png'))
    plt.close()

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    print("正在评估模型...")
    start_time = time.time()

    # 分批预测以节省内存
    batch_size = 1000
    y_pred_list = []  # 改用列表存储预测结果

    for i in tqdm(range(0, len(X_test), batch_size), desc="预测进度"):
        batch_x = X_test[i:i + batch_size]
        batch_pred = model.predict(batch_x)
        y_pred_list.extend(batch_pred)

    y_pred = np.array(y_pred_list)  # 转换为numpy数组

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    evaluation_time = time.time() - start_time
    print(f"评估用时: {evaluation_time:.2f}秒")

    # 确保返回三个值
    return float(accuracy), report, y_pred

def save_training_info(output_dir, accuracy, report, train_time=None):
    """保存训练信息"""
    info_path = os.path.join(output_dir, 'training_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if train_time is not None:
            f.write(f"训练用时: {train_time:.2f}秒\n")
        f.write(f"\n模型准确率: {accuracy:.4f}\n\n")
        f.write("分类报告:\n")
        f.write(report)

def main():
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    # 检查MPS可用性
    print(f"MPS是否可用: {torch.backends.mps.is_available()}")

    # 加载完整数据集
    X_train, X_test, y_train, y_test = load_mnist_data()

    # 创建模型
    print("正在创建模型...")
    model = SVMClassifier(kernel='rbf')

    train_time = None
    # 尝试加载已保存的模型
    if not model.load_model():
        print("开始训练模型...")
        train_start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start_time
        print(f"模型训练完成，用时: {train_time:.2f}秒")

    # 评估模型
    accuracy, report, y_pred = evaluate_model(model, X_test, y_test)

    print(f"\n模型准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(report)

    # 保存评估报告和训练信息
    save_training_info(model.output_dir, accuracy, report, train_time)

    # 绘制混淆矩阵
    print("正在绘制混淆矩阵...")
    plot_confusion_matrix(y_test, y_pred, model.output_dir)

    # 展示预测示例
    print("正在绘制预测示例...")
    plot_sample_predictions(X_test, y_test, y_pred, model.output_dir)

    print(f"\n所有结果已保存到目录: {model.output_dir}")

if __name__ == "__main__":
    main()