#batch_main_unet_recog_cvl.py
#
#和CVL版的 batch_main_recog2.py 是同一件事：读PNG图片文件夹，跑一遍，存CSV、
#打印整体准确率。区别是把 LeftRightRecognizer（走廊+路径搜索那一套）换成
#UNetLeftRightRecognizerCVL（直接复用TDTS训练出来的U-Net权重，zero-shot跑CVL）。
#
#跑之前确认：
#   - unet_tdts.pt 已经用 unet_seg_train.py 在TDTS上训练好（这里不重新训练，
#     CVL没有逐像素真值掩码，没法做同样的监督训练，本来就是要测zero-shot泛化）
#   - CNN_MODEL_PATH 换成你CVL识别模型实际路径
#   - DATA_ROOT 换成你PNG数据实际所在目录

import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
import glob
from PIL import Image
from pathlib import Path
from unet_left_right_recog_cvl import UNetLeftRightRecognizerCVL

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / 'data' / 'cvl_double_char'   # TODO: 换成你实际的CVL PNG图片目录
CNN_MODEL_PATH = PROJECT_ROOT / 'models' / 'cvl_cnn_improved.pth'
UNET_WEIGHT_PATH = PROJECT_ROOT / 'models' / 'unet_tdts.pt'   # zero-shot：直接用TDTS训练出来的权重，不额外微调

USE_DENOISE = True

class PNGDigitDataset(Dataset):
    """从PNG文件读取数字图片，标签从文件名解析（第一个'-'前的数字）"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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

def main():
    transform = transforms.Compose([transforms.ToTensor()])
    img_data = PNGDigitDataset(root_dir=DATA_ROOT, transform=transform)
    total_samples = len(img_data)
    print(f"CVL PNG数据集总样本数: {total_samples}")

    recognizer = UNetLeftRightRecognizerCVL(CNN_MODEL_PATH, UNET_WEIGHT_PATH, use_denoise=USE_DENOISE)
    print("U-Net(TDTS训练,zero-shot) + CVL CNN识别器加载完成")

    rows, accuracy = recognizer.process_batch_images_unet(img_data)

    os.makedirs('outPutCVlTest', exist_ok=True)
    df = pd.DataFrame(rows, columns=['ImgID', 'TrueLbl', 'UNetPred', 'OK', 'LeftConf', 'RightConf'])
    out_path = os.path.join('outPutCVlTest', 'unet_cvl_zeroshot_results.csv')
    df.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"\n=== U-Net (TDTS训练, zero-shot) 在CVL上的整体识别准确率 ===")
    print(f"N={total_samples}  Acc={accuracy*100:.2f}%  结果已存到 {out_path}")
    print("可以直接对照 Table 11 里其它方法 TDTS-TS -> CVL 的跨域衰减幅度。")


if __name__ == '__main__':
    main()
