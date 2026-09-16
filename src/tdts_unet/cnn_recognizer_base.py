#cnn_recognizer_base.py
#
#从 u_left_right_recog1.py 里把"CNN结构 + 加载模型 + 识别单张/批量图片"这部分原样抽出来，
#不包含任何 cut_region.py / path_generationSafe10.py 相关的导入。
#
#目的：让 U-Net 这条路径彻底不依赖切割域/路径搜索的代码，纯粹是"复用同一个训练好的识别模型"，
#而不是"继承一个内部还夹带着图论方法的类"。对图论方法(u_left_right_recog1.py)本身不做任何改动，
#避免影响你已经跑通、验证过的那套流水线。

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image


class CNN(nn.Module):
    """和 u_left_right_recog1.py / cnn4032noise1dig3.py 里完全一样的结构，
    输入32x40单通道图，输出10类(0-9)logits。"""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 8 * 10, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 10)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


class BaseCharRecognizer:
    """只负责"加载CNN + 识别单张/批量32x40图片"，不涉及任何分割逻辑。
    图论方法(LeftRightRecognizer)和U-Net方法(UNetLeftRightRecognizer)都可以继承它，
    各自只需要实现自己的分割步骤。"""

    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _load_model(self, model_path):
        model = CNN()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def recognize_image(self, image_array):
        image_pil = Image.fromarray(image_array.astype('uint8'))
        image_tensor = self.preprocess(image_pil).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item()
        return pred, confidence

    def recognize_batch(self, image_list):
        if not image_list:
            return [], []
        batch_tensors = [self.preprocess(Image.fromarray(img.astype('uint8'))) for img in image_list]
        batch_tensor = torch.stack(batch_tensors)
        with torch.no_grad():
            output = self.model(batch_tensor)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            confidences = probs[torch.arange(probs.size(0)), preds]
        return preds.tolist(), confidences.tolist()
