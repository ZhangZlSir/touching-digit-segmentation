#batch_main_unet_recog.py
#
#
#跑法（先确认前置步骤都做完）：
#   1. python mask3240_gen.py         生成 test-masks.npy
#   2. python unet_seg_train.py       训练 U-Net，得到 ./unet_tdts.pt
#   3. python batch_main_unet_recog.py   本脚本：跑分割+识别，输出整体准确率
#


import os
import pandas as pd
from torchvision import datasets, transforms
from pathlib import Path

from unet_left_right_recog import UNetLeftRightRecognizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_img  = PROJECT_ROOT / 'data' / 'tdts_dataset' 
DATA_model = PROJECT_ROOT / 'models' 

CNN_MODEL_PATH = DATA_model / 'tdts_mrm.pth'   

UNET_WEIGHT_PATH = DATA_model / 'unet_tdts.pt'
USE_DENOISE = True


def run_split(recognizer, dataset, split_name, output_dir='outPutTest'):
    os.makedirs(output_dir, exist_ok=True)
    rows, accuracy = recognizer.process_batch_images_unet(dataset)

    df = pd.DataFrame(rows, columns=['ImgID', 'TrueLbl', 'UNetPred', 'OK', 'LeftConf', 'RightConf'])
    out_path = os.path.join(output_dir, f'unet_{split_name}_results.csv')
    df.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"[{split_name}] 整体识别准确率: {accuracy:.4f}  ({df['OK'].sum()}/{len(df)})，结果已存到 {out_path}")
    return accuracy


def main():
    transform = transforms.Compose([transforms.ToTensor()])

    # （idx-ubyte 命名和标准 MNIST 一致，torchvision 的 MNIST loader 直接能读）
    train_set = datasets.MNIST(DATA_img, download=False, transform=transform, train=True)
    test_set = datasets.MNIST(DATA_img, download=False, transform=transform, train=False)

    recognizer = UNetLeftRightRecognizer(CNN_MODEL_PATH, UNET_WEIGHT_PATH, use_denoise=USE_DENOISE)
    print("U-Net + CNN 识别器加载完成")

    acc_tr = run_split(recognizer, train_set, 'TDTS-TR')
    acc_ts = run_split(recognizer, test_set, 'TDTS-TS')

    print("\n=== 汇总（可以直接填进 Table 4 新增的 U-Net 列）===")
    print(f"MRM  U-Net   TDTS-TR: {acc_tr*100:.2f}%")
    print(f"MRM  U-Net   TDTS-TS: {acc_ts*100:.2f}%")


if __name__ == '__main__':
    main()
