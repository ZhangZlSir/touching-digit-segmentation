#cnn4032-noise1dig3.py
#本程序用于识别一个带噪声的数字图片是0-9的哪个。,训练后做了训练模型的保存工作
#同时保存2种格式。使用的库包含了有噪声的和无噪声的数字图片
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import csv
from matplotlib import pyplot as plt
import os
import time

# 创建模型保存目录

from pathlib import Path

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层，输入通道数为1（灰度图像），输出通道数为16，卷积核大小为3x3，padding为1保持图像尺寸不变
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # ReLU激活函数，增加模型的非线性
        self.relu1 = nn.ReLU()
        # 最大池化层，池化核大小为2x2，用于下采样减少计算量和防止过拟合
        self.pool1 = nn.MaxPool2d(2)
        # 第二个卷积层，输入通道数为16，输出通道数为32，卷积核大小为3x3，padding为1保持图像尺寸不变
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # ReLU激活函数
        self.relu2 = nn.ReLU()
        # 最大池化层，池化核大小为2x2
        self.pool2 = nn.MaxPool2d(2)
        # 第一个全连接层，输入大小为32 * 8 * 10（经过两次卷积和池化后的特征图大小），输出大小为128
        self.fc1 = nn.Linear(32 * 8 * 10, 128)
        # ReLU激活函数
        self.relu3 = nn.ReLU()
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.5)
        # 第二个全连接层，输入大小为128，输出大小为10（对应10个数字类别）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 前向传播过程，依次经过卷积、激活、池化操作
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # 将特征图展平为一维向量，以便输入到全连接层
        x = x.view(-1, 32 * 8 * 10)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 数据预处理
transform = transforms.Compose([
    # 将图像转换为PyTorch张量
    transforms.ToTensor(),
    # 对图像进行归一化，使用更通用的参数
    transforms.Normalize((0.5,), (0.5,))
])
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / 'saved_models'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)     # 不存在则创建
#DATA_clean = PROJECT_ROOT / 'data' / 'tdts_single_char' / 'clean'
DATA_mixed = PROJECT_ROOT / 'data' / 'tdts_single_char' / 'mixed'
#DATA_noisy = PROJECT_ROOT / 'data' / 'tdts_single_char' / 'noisy'
# 加载数据集
# 从'./data6'目录加载训练集数据，应用数据预处理，不重新下载数据
train_dataset = MNIST(root=DATA_mixed, train=True, transform=transform, download=False)
# 从'./data6'目录加载测试集数据，应用数据预处理，不重新下载数据
test_dataset = MNIST(root=DATA_mixed, train=False, transform=transform, download=False)

# 创建数据加载器 - 增加批次大小到128
# 训练数据加载器，批次大小为128，打乱数据顺序
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
# 测试数据加载器，批次大小为128，不打乱数据顺序
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

# 初始化模型、损失函数和优化器
# 实例化定义的CNN模型
model = CNN()
# 选择交叉熵损失函数，用于计算模型输出与真实标签之间的损失
criterion = nn.CrossEntropyLoss()
# 选择Adam优化器，用于更新模型的参数，学习率为0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 添加学习率调度器，每10个epoch将学习率降低为原来的0.5
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 训练参数
num_epochs = 25
# 用于存储每个epoch的训练损失
train_losses = []
# 用于存储每个epoch的训练准确率
train_accuracies = []
# 用于存储每个epoch的测试损失
test_losses = []
# 用于存储每个epoch的测试准确率
test_accuracies = []

# 记录开始时间
start_time = time.time()

# 训练循环
for epoch in range(num_epochs):
    # 设置模型为训练模式
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    # 遍历训练数据加载器中的每个批次
    for images, labels in train_loader:
        # 梯度清零，防止梯度累加
        optimizer.zero_grad()
        # 前向传播，计算模型输出
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播，计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()

        # 累加训练损失
        train_loss += loss.item()
        # 获取预测的类别索引
        _, predicted = torch.max(outputs.data, 1)
        # 累加总样本数
        total_train += labels.size(0)
        # 累加正确预测的样本数
        correct_train += (predicted == labels).sum().item()

    # 更新学习率
    scheduler.step()
    
    # 计算平均训练损失
    train_loss = train_loss / len(train_loader)
    # 计算训练准确率
    train_accuracy = 100 * correct_train / total_train
    # 将训练损失和准确率添加到相应列表中
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 测试循环
    model.eval()
    test_loss = 0
    correct_test = 0
    total_test = 0

    # 不计算梯度，减少内存消耗和计算量
    with torch.no_grad():
        # 遍历测试数据加载器中的每个批次
        for images, labels in test_loader:
            # 前向传播，计算模型输出
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, labels)

            # 累加测试损失
            test_loss += loss.item()
            # 获取预测的类别索引
            _, predicted = torch.max(outputs.data, 1)
            # 累加总样本数
            total_test += labels.size(0)
            # 累加正确预测的样本数
            correct_test += (predicted == labels).sum().item()

    # 计算平均测试损失
    test_loss = test_loss / len(test_loader)
    # 计算测试准确率
    test_accuracy = 100 * correct_test / total_test
    # 将测试损失和准确率添加到相应列表中
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # 打印当前学习率
    current_lr = scheduler.get_last_lr()[0]
    print(f'Epoch {epoch + 1}/{num_epochs}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

    # 从第13轮开始保存模型
    if epoch >= 12:
        # 保存完整模型
        torch.save(model, OUTPUT_DIR / 'digit1_modelCln_epoch_{epoch+1}.pth')
        # 保存模型状态字典
        torch.save(model.state_dict(), OUTPUT_DIR / 'digit1_modelCln_state_epoch_{epoch+1}.pth')
        print(f"模型已保存: {OUTPUT_DIR / 'digit1_modelCln_epoch_{epoch+1}.pth'}")

# 计算总训练时间
end_time = time.time()
total_time = end_time - start_time
print(f"训练完成！总耗时: {total_time//60:.0f}分 {total_time%60:.0f}秒")

# 保存最终模型
torch.save(model, OUTPUT_DIR / 'digit1_recog_modelCln.pth')
torch.save(model.state_dict(), OUTPUT_DIR / 'digit1_recogDict_modelCln.pth')

# 保存数据到 CSV 文件
with open(OUTPUT_DIR / 'trainingCln_stats.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入CSV文件的表头
    writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])
    for i in range(num_epochs):
        # 写入每个epoch的训练和测试数据
        writer.writerow([i + 1, train_losses[i], train_accuracies[i], test_losses[i], test_accuracies[i]])

# 绘制训练和测试损失曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
# 绘制训练损失曲线
plt.plot(train_losses, label='Train Loss')
# 绘制测试损失曲线
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss')

# 绘制训练和测试准确率曲线
plt.subplot(1, 2, 2)
# 绘制训练准确率曲线
plt.plot(train_accuracies, label='Train Accuracy')
# 绘制测试准确率曲线
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Test Accuracy')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'trainingCln_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印最佳准确率
best_train_acc = max(train_accuracies)
best_test_acc = max(test_accuracies)
best_train_epoch = train_accuracies.index(best_train_acc) + 1
best_test_epoch = test_accuracies.index(best_test_acc) + 1

print(f"最佳训练准确率: {best_train_acc:.2f}% (第 {best_train_epoch} 轮)")
print(f"最佳测试准确率: {best_test_acc:.2f}% (第 {best_test_epoch} 轮)")