#cnn_recognizer_base_cvl.py
#
#从 u_left_right_recog1_cvl.py 里把"CVL版CNN结构 + 加载模型 + 识别单张/批量图片"
#这部分原样抽出来，不带任何 cut_region.py / path_generationSafe11.py / corridor_utils.py
#的导入——原因和 TDTS 那边的 cnn_recognizer_base.py 完全一样：U-Net这条路径不需要
#切割域/走廊/路径搜索的代码，只需要"同一个训练好的CVL识别模型"。
#
#注意：CVL版CNN结构和TDTS版不一样（3层卷积+BN，见下面CNN类），checkpoint也不是
#同一个文件，所以不能直接复用 cnn_recognizer_base.py，要单独建一份。

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class CNN(nn.Module):
    """与 u_left_right_recog1_cvl.py / cnn4032-cvl-improved.py 完全相同的结构"""
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

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()                # -> 64 x 8 x 10

        self.flatten_dim = 64 * 8 * 10
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_p * 0.4)
        self.dropout2 = nn.Dropout(dropout_p)
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


class BaseCharRecognizerCVL:
    """只负责"加载CVL版CNN + 识别单张/批量图片"，不涉及任何分割/走廊逻辑。"""

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
