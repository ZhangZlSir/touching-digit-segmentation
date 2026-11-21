#verify_pairs_match_images.py
# 数据一致性检查：用配对表 + 坐标重建「无噪声字符层」，再与 data/tdts_dataset 中的图像逐像素比对。
#
# 原理（与生成脚本 s4_build_tdts.py 的顺序一致）：
#   1. 先撒背景噪点；2. 再把左字符 paste 到 (0,6)、右字符按 (x0,y0) 贴上去。
# 因此凡是字符覆盖到的像素，库图中的值必然等于重建值，与随机噪声无关；若配对表或坐标不匹配，
# 字符层会整体对不上。可用于确认数据集与 data/tdts_build_inputs/ 下的配对表、坐标是否配套。
#
# 用法：
#   python verify_pairs_match_images.py                 # 训练集抽样 2000 + 测试集全量
#   python verify_pairs_match_images.py --samples 0     # 全量比对（较慢）
#   python verify_pairs_match_images.py --pairs-test <另一份配对表.csv>

import argparse
import csv
import io
import os
import struct
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_IN = PROJECT_ROOT / 'data' / 'mnist_compact_20x20' / 'MNIST' / 'raw'
DATA_TDTS = PROJECT_ROOT / 'data' / 'tdts_dataset' / 'MNIST' / 'raw'
BUILD_INPUTS = PROJECT_ROOT / 'data' / 'tdts_build_inputs'


def load_idx_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)


def load_idx_labels(path):
    with open(path, 'rb') as f:
        struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_csv(path):
    with io.open(path, newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def predict_character_layer(left, right, x0, y0):
    """按 s4_build_tdts.py 的融合规则重建无噪声字符层，并返回被字符覆盖的像素掩码。"""
    pred = np.zeros((32, 40), dtype=np.uint8)
    cover = np.zeros((32, 40), dtype=bool)
    pred[6:26, 0:20] = left
    cover[6:26, 0:20] = True
    for ii in range(20):
        for j in range(20):
            x, y = x0 + ii, y0 + j
            if 0 <= x < 40 and 0 <= y < 32:
                if 0 <= x < 20 and 6 <= y < 26:
                    pred[y, x] = max(int(pred[y, x]), int(right[j, ii]))
                else:
                    pred[y, x] = right[j, ii]
                cover[y, x] = True
    return pred, cover


def verify(tag, chars, images, labels, pairs, coords, samples):
    n = len(pairs) if samples in (0, None) else min(samples, len(pairs))
    step = max(1, len(pairs) // n) if samples else 1
    idxs = list(range(0, len(pairs), step))[:n] if samples else list(range(len(pairs)))
    bad_px = total_px = bad_img = bad_lbl = 0
    for i in idxs:
        p, c = pairs[i], coords[i]
        left = chars[int(p['Left Image Index'])]
        right = chars[int(p['Right Image Index'])]
        pred, cover = predict_character_layer(left, right, int(c['x']), int(c['y']))
        diff = int((images[i][cover] != pred[cover]).sum())
        bad_px += diff
        total_px += int(cover.sum())
        bad_img += (diff > 0)
        expected = int(p['Left Label']) * 10 + int(p['Right Label'])
        if expected < 10:
            expected += 100
        bad_lbl += (int(labels[i]) != expected)
    pct = bad_px / total_px * 100 if total_px else 0.0
    print(f"{tag}: 比对 {len(idxs)} 张 | 不一致像素 {bad_px}/{total_px} ({pct:.5f}%) | "
          f"有差异的图像 {bad_img} | 标签不符 {bad_lbl}")
    return bad_px, bad_lbl


def parse_args():
    p = argparse.ArgumentParser(description='Check which pair table generated the TDTS images')
    p.add_argument('--samples', type=int, default=2000, help='每份配对表抽样数；0 表示全量')
    p.add_argument('--pairs-test', type=str, default=None, help='额外的测试集配对表（用于对比）')
    return p.parse_args()


def main():
    args = parse_args()
    chars_tr = load_idx_images(DATA_IN / 'train-images-idx3-ubyte')
    chars_te = load_idx_images(DATA_IN / 't10k-images-idx3-ubyte')
    tdts_tr = load_idx_images(DATA_TDTS / 'train-images-idx3-ubyte')
    tdts_te = load_idx_images(DATA_TDTS / 't10k-images-idx3-ubyte')
    lbl_tr = load_idx_labels(DATA_TDTS / 'train-labels-idx1-ubyte')
    lbl_te = load_idx_labels(DATA_TDTS / 't10k-labels-idx1-ubyte')
    coords_tr = load_csv(BUILD_INPUTS / 'coordinates_train.csv')
    coords_te = load_csv(BUILD_INPUTS / 'coordinates_tst.csv')

    verify('TRAIN  seed42', chars_tr, tdts_tr, lbl_tr,
           load_csv(BUILD_INPUTS / 'mnist_image_pairs_train.csv'), coords_tr, args.samples)
    verify('TEST   seed42', chars_te, tdts_te, lbl_te,
           load_csv(BUILD_INPUTS / 'mnist_image_pairs_tst.csv'), coords_te, args.samples)
    if args.pairs_test:
        verify('TEST   ' + os.path.basename(args.pairs_test), chars_te, tdts_te, lbl_te,
               load_csv(args.pairs_test), coords_te, args.samples)


if __name__ == '__main__':
    main()
