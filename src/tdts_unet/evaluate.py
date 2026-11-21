#evaluate.py
#
#
#跑法（先确认前置步骤都做完）：
#   1. python build_masks.py    生成 data/tdts_masks/{train,test}-masks.npy
#   2. python train_unet.py     训练 U-Net，得到 models/unet_tdts.pt
#   3. python evaluate.py       本脚本：跑分割+识别，输出整体准确率
#


import os
import pandas as pd
from torchvision import datasets, transforms
from pathlib import Path

from pipeline import UNetLeftRightRecognizer

import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_img  = PROJECT_ROOT / 'data' / 'tdts_dataset' 
DATA_model = PROJECT_ROOT / 'models' 

MODEL_PATHS = {name: DATA_model / f'tdts_{name}.pth' for name in ('crm', 'nrm', 'mrm')}


def parse_args():
    p = argparse.ArgumentParser(description='U-Net segmentation + recognition on TDTS')
    p.add_argument('--model', choices=('crm', 'nrm', 'mrm'), default='mrm', help='downstream recognition model')
    p.add_argument('--split', choices=('tr', 'ts'), default=None, help='only run one split; default runs both')
    return p.parse_args()
   

UNET_WEIGHT_PATH = DATA_model / 'unet_tdts.pt'
USE_DENOISE = True


def run_split(recognizer, dataset, split_name, output_dir=None):
    output_dir = output_dir or (Path(__file__).resolve().parent / 'outPutTest')
    os.makedirs(output_dir, exist_ok=True)
    rows, accuracy = recognizer.process_batch_images_unet(dataset)

    df = pd.DataFrame(rows, columns=['ImgID', 'TrueLbl', 'UNetPred', 'OK', 'LeftConf', 'RightConf'])
    out_path = os.path.join(output_dir, f'unet_{split_name}_results.csv')
    df.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"[{split_name}] 整体识别准确率: {accuracy:.4f}  ({df['OK'].sum()}/{len(df)})，结果已存到 {out_path}")
    return accuracy


def main():
    args = parse_args()
    transform = transforms.Compose([transforms.ToTensor()])

    # （idx-ubyte 命名和标准 MNIST 一致，torchvision 的 MNIST loader 直接能读）
    train_set = datasets.MNIST(DATA_img, download=False, transform=transform, train=True)
    test_set = datasets.MNIST(DATA_img, download=False, transform=transform, train=False)

    recognizer = UNetLeftRightRecognizer(MODEL_PATHS[args.model], UNET_WEIGHT_PATH, use_denoise=USE_DENOISE)
    print("U-Net + CNN 识别器加载完成")

    acc_tr = run_split(recognizer, train_set, 'TDTS-TR') if args.split in (None, 'tr') else None
    acc_ts = run_split(recognizer, test_set, 'TDTS-TS') if args.split in (None, 'ts') else None

    print("\n=== 汇总（可以直接填进 Table 4 新增的 U-Net 列）===")
    for _tag, _acc in (("TDTS-TR", acc_tr), ("TDTS-TS", acc_ts)):
        if _acc is not None:
            print(f"{args.model.upper():4s} U-Net   {_tag}: {_acc*100:.2f}%")


if __name__ == '__main__':
    main()
