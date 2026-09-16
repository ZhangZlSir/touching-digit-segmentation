# cnn4032-cvl-improved.py
# 在原 cnn4032-cvl.py 基础上做的改进版，目标：提升验证/测试集准确率、缓解过拟合。
#
# 相对原版的主要改动（详见每处注释 "【改动】"）：
#   1. 训练集做数据增强（小幅旋转/平移/缩放 + 随机擦除），测试集不增强。
#   2. 网络加入 BatchNorm，多加一层卷积，增大特征容量的同时用 BN 帮助收敛更稳。
#   3. Adam 换成 AdamW 并加 weight_decay，减小过拟合。
#   4. 学习率调度由固定 StepLR 换成 ReduceLROnPlateau，按验证集 loss 自适应下降。
#   5. 引入独立的验证集（从训练集里划出一部分），测试集只在最后评估一次，
#      避免"用测试集调超参/选模型"这种隐性的数据泄漏。
#   6. 只保存"验证集准确率最高"的一份模型，而不是从第13轮开始每轮都存。
#   7. 训练结束后额外输出混淆矩阵/每类准确率，方便定位是哪些数字容易混淆。
#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import csv
from matplotlib import pyplot as plt
import os
import time
from PIL import Image
import glob
from pathlib import Path

# -------------------- 自定义数据集类 --------------------
class PNGDigitDataset(Dataset):
    """从PNG文件读取数字图片，标签从文件名解析（第一个'-'前的数字）"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if not os.path.isdir(root_dir):
            raise RuntimeError(
                f"Directory does not exist: {root_dir}\n"
                f"Please download and extract the dataset to this path."
            )
        self.image_paths = glob.glob(os.path.join(root_dir, '*.png')) 
        if not self.image_paths:
            raise RuntimeError(f"在 {root_dir} 中没有找到PNG图片")

        self.labels = []
        for path in self.image_paths:
            fname = os.path.basename(path)
            label_str = fname.split('-')[0]
            if not label_str.isdigit():
                raise ValueError(f"无法从文件名 {fname} 解析数字标签")
            self.labels.append(int(label_str))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# 【改动】给基础数据集包一层，允许训练/验证使用不同的 transform
# （因为 random_split 出来的 Subset 默认共享同一个底层 dataset 的 transform）
class TransformWrapper(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image_path_dataset = self.subset.dataset
        real_idx = self.subset.indices[idx]
        img_path = image_path_dataset.image_paths[real_idx]
        label = image_path_dataset.labels[real_idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label


# -------------------- 模型定义 --------------------
class CNN(nn.Module):
    """
    【改动】相对原模型：
      - 每个卷积后加 BatchNorm2d，训练更稳定、收敛更快，通常也更抗过拟合。
      - 增加第三个卷积块（不downsample，只提特征），提升表达能力。
      - fc1 前加一次 dropout（原来只在 fc1 之后有），双重正则化。
    输入尺寸假设仍为 1x32x40（高 x 宽）。
    """
    def __init__(self, num_classes=10, dropout_p=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)          # -> 16 x 16 x 20

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)          # -> 32 x 8 x 10

        # 【改动】新增卷积块，不做下采样，只加深特征提取
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.flatten_dim = 64 * 8 * 10
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_p * 0.4)   # 卷积输出后的轻量 dropout
        self.dropout2 = nn.Dropout(dropout_p)         # 原来的 dropout
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = x.view(-1, self.flatten_dim)
        x = self.dropout1(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / len(loader), 100 * correct / total, all_preds, all_labels


# -------------------- 主程序 --------------------
if __name__ == '__main__':
    os.makedirs('saved_modelsCvl_improved', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}") 

    torch.manual_seed(42)
    np.random.seed(42)

    # ---------- 数据预处理 ----------
    # 训练集：加入轻微的仿射增强（旋转/平移/缩放）和随机擦除，
    # 幅度要小——数字类别本身对角度、位置比较敏感，增强太猛反而会把 6/9、2/7 之类的形变成错的类。
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=6, translate=(0.06, 0.06), scale=(0.92, 1.08)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
    ])
    # 验证/测试集：不做增强，只做和训练一致的归一化
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    IMG_DIR = PROJECT_ROOT / 'cvl_single_char'

    # ---------- 加载数据集 ----------
    train_root = IMG_DIR / 'trainStd40-32'   # 训练集目录
    test_root  = IMG_DIR / 'evalStd40-32'    # 测试集目录（改动后：仅用于最终评估，不参与调参/选模型）

    full_train_dataset = PNGDigitDataset(root_dir=train_root, transform=None)
    test_dataset = PNGDigitDataset(root_dir=test_root, transform=eval_transform)

    # 【改动】从训练集中划出 10% 作为验证集，用于早停/学习率调度/选择最佳模型，
    # 测试集只在训练全部结束后跑一次，得到真正无偏的最终指标。
    val_ratio = 0.1
    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size
    train_subset, val_subset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_dataset = TransformWrapper(train_subset, train_transform)
    val_dataset = TransformWrapper(val_subset, eval_transform)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}（从训练集中划出，用于选模型/调LR）")
    print(f"测试集样本数: {len(test_dataset)}（仅用于最终评估）")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # ---------- 初始化模型、损失函数、优化器 ----------
    model = CNN().to(device)
    # 【改动】label smoothing 减轻模型对训练标签"过度自信"，对提升泛化常有帮助
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # 【改动】AdamW + weight_decay 做 L2 正则化，减小过拟合
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # 【改动】按验证集loss自适应降学习率，比固定StepLR更贴合实际收敛情况
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ---------- 训练参数 ----------
    num_epochs = 40                 # 配合早停，可以适当调大轮数上限
    early_stop_patience = 8         # 验证集准确率连续多少轮不提升就停
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_path = 'saved_modelsCvl_improved/digit1_modelCvl_best.pth'

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch+1}/{num_epochs}, LR: {current_lr:.6f}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

        # 【改动】只保存验证集上表现最好的一份模型，而不是从某轮开始每轮都存
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> 验证集准确率提升，已保存最佳模型: {best_model_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"验证集准确率连续 {early_stop_patience} 轮未提升，提前停止训练。")
            break

    end_time = time.time()
    total_time = end_time - start_time
    print(f"训练完成！总耗时: {total_time//60:.0f}分 {total_time%60:.0f}秒")

    # ---------- 用最佳模型在测试集上做最终评估 ----------
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_accuracy, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"最终测试集（最佳验证模型）: Loss: {test_loss:.4f}, Acc: {test_accuracy:.2f}%")

    # 保存最终模型（完整 + state_dict）
    torch.save(model, 'digit1_recog_modelCvl_improved.pth')
    torch.save(model.state_dict(), 'digit1_recogDict_modelCvl_improved.pth')

    # ---------- 保存训练统计到CSV ----------
    with open('trainingCvl_improved_stats.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy'])
        for i in range(len(train_losses)):
            writer.writerow([i + 1, train_losses[i], train_accuracies[i], val_losses[i], val_accuracies[i]])

    # ---------- 绘制曲线 ----------
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig('trainingCvl_improved_curves.png', dpi=300, bbox_inches='tight')

    # ---------- 每个数字类别的准确率（定位容易混淆的数字） ----------
    print("\n各数字类别在测试集上的准确率：")
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    for digit in range(10):
        mask = test_labels == digit
        if mask.sum() > 0:
            acc = 100 * (test_preds[mask] == test_labels[mask]).sum() / mask.sum()
            print(f"  数字 {digit}: {acc:.2f}%  (样本数 {mask.sum()})")

    best_train_acc = max(train_accuracies)
    best_val_acc_print = max(val_accuracies)
    print(f"\n最佳训练准确率: {best_train_acc:.2f}%")
    print(f"最佳验证准确率: {best_val_acc_print:.2f}%")
    print(f"最终测试准确率: {test_accuracy:.2f}%")