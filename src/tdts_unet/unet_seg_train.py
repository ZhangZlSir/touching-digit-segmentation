#unet_seg_train.py
#
#训练 U-Net 分割模型，和现有的 A*/Dijkstra/VPM/WA 走同一套预处理（去噪，denoising1.py），
#这样比较时差异只来自"分割方法本身"，不掺杂预处理不一致的干扰。
#
#依赖：
#   mask3240_gen.py 先跑一遍，生成 train-masks.npy 和 test-masks.npy
#   
#   denoising1.py 里的 remove_salt_pepper_noise，和其它方法用的是同一个函数
#
#用法：
#   python unet_seg_train.py
#产出：
#   ./unet_tdts.pt          U-Net 权重
#   每个epoch打印 train_loss 和 test集上的 mean IoU / 逐类IoU

import struct
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from unet_seg_model import TinyUNet
from denoising1 import remove_salt_pepper_noise
from pathlib import Path


def read_idx_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        buf = f.read(rows * cols * num)
        data = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)
    return data  # (N, 32, 40)


class TDTSSegDataset(Dataset):
    def __init__(self, image_idx_path, mask_npy_path, use_denoise=True):
        raw_images = read_idx_images(image_idx_path)
        self.masks = np.load(mask_npy_path)
        assert len(raw_images) == len(self.masks), \
            "图像和掩码数量对不上：确认 mask3240_gen.py 用的是和生成 data17 时同一份 csv"

        if use_denoise:
            # 和其它分割方法保持同一套预处理，逐张跑一遍去噪（numba jit过，单进程也不算慢）
            self.images = np.stack(
                [remove_salt_pepper_noise(img, True) for img in raw_images], axis=0
            )
        else:
            self.images = raw_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        mask = self.masks[idx].astype(np.int64)
        img_t = torch.from_numpy(img).unsqueeze(0)  # (1,32,40)
        mask_t = torch.from_numpy(mask)              # (32,40)
        return img_t, mask_t


def compute_class_weights(mask_npy_path, num_classes=3):
    masks = np.load(mask_npy_path)
    counts = np.bincount(masks.reshape(-1), minlength=num_classes).astype(np.float64)
    freq = counts / counts.sum()
    weights = 1.0 / np.clip(freq, 1e-6, None)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def per_class_iou(pred, target, num_classes=3):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        ious.append(inter / union if union > 0 else float('nan'))
    return ious

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_img  = PROJECT_ROOT / 'data' / 'tdts_dataset'/ 'MNIST' / 'RAW' 
DATA_mask = PROJECT_ROOT / 'data' / 'tdts_masks'
DATA_output = PROJECT_ROOT / 'models' 
DATA_output.mkdir(parents=True, exist_ok=True)     # 不存在则创建

def train(
    train_img_path=DATA_img / 'train-images-idx3-ubyte',#其他文件生成的，在上级目录
    train_mask_path=DATA_mask / 'train-masks.npy',#本项目生成的，在同级目录
    test_img_path=DATA_img / 't10k-images-idx3-ubyte',
    test_mask_path=DATA_mask / 'test-masks.npy',
    use_denoise=True,
    epochs=15,
    batch_size=128,
    lr=1e-3,
    save_path=DATA_output / 'unet_tdts.pt',
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"训练设备: {device}")

    if save_path.exists() and save_path.is_file():
        print(f"Checkpoint found at {save_path}, skipping training.")
        return

    train_ds = TDTSSegDataset(train_img_path, train_mask_path, use_denoise)
    test_ds = TDTSSegDataset(test_img_path, test_mask_path, use_denoise)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(train_mask_path).to(device)
    print(f"类别权重(背景/左/右): {class_weights.tolist()}")

    model = TinyUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_miou = -1.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
        scheduler.step()
        avg_loss = total_loss / len(train_ds)

        model.eval()
        all_ious = []
        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                pred = model(imgs).argmax(dim=1)
                all_ious.append(per_class_iou(pred.cpu(), masks.cpu()))
        all_ious = np.array(all_ious, dtype=np.float64)
        cls_iou = np.nanmean(all_ious, axis=0)
        miou = np.nanmean(cls_iou)

        print(f"epoch {epoch+1}/{epochs}  train_loss={avg_loss:.4f}  "
              f"mIoU={miou:.4f}  IoU(bg/left/right)={cls_iou.round(4).tolist()}")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), save_path)
            print(f"  -> 新的最佳 mIoU，已保存到 {save_path}")

    print(f"训练完成，最佳 mIoU={best_miou:.4f}，权重在 {save_path}")


if __name__ == '__main__':
    train()
