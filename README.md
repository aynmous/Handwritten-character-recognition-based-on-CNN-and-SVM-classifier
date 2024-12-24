# Handwritten-character-recognition-based-on-CNN-and-SVM-classifier
Handwritten character recognition based on CNN and SVM classifier
# 项目名称

基于CNN特征提取与SVM分类器的混合算法在手写字符识别中的应用研究。

## 环境要求

- Python 3.8
- 需要的包：
  - numpy
  - pandas
  - matplotlib

## 安装步骤

1. 安装所需的包：
   pip install -r requirements.txt
2. 运行SVM算法：
   python SVM.py
3. 运行CNN_SVM算法：
   python CNN_SVM.py

## 实验结果保存

### 文件保存路径

实验结果将保存在以下目录结构中：results-SVM/
│
├── run_20241121_165532/
│   ├── confusion_matrix.png        # 混淆矩阵的可视化图片
│   ├── sample_predictions.png      # 示例预测结果图片
│   ├── scaler.joblib               # 标准化器模型文件
│   ├── svm_model.joblib            # SVM模型文件
│   └── training_info.txt           # 训练过程的相关信息
│
results_CNN_SVM/
│
└── 20241121_204246/
├── adaboost_model.joblib       # Adaboost模型文件
├── adaboost_scaler.joblib      # Adaboost模型的标准化器文件
├── confusion_matrix.png        # 混淆矩阵的可视化图片
├── evaluation_report.txt       # 模型评估报告
├── learning_curve_simple.png   # 学习曲线的可视化图片
└── sample_predictions.png      # 示例预测结果图片


## 注意事项

- **运行环境**：本项目使用 **macOS** 运行，支持 **Apple Silicon（M1/M2）** 芯片。
- **加速方式**：项目中利用 PyTorch 的 **Metal Performance Shaders（MPS）** 加速进行深度学习模型的训练与推理。在设备检测时，自动启用 `mps` 作为加速设备。
- **设备优先级**：如果运行设备不支持 MPS，代码将自动切换到 **CPU** 或 **CUDA**（若可用）。

