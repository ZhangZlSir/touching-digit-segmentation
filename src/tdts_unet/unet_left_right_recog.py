#unet_left_right_recog.py
#
#和之前版本的唯一区别：不再 import u_left_right_recog1.py 里的 LeftRightRecognizer
#（那条导入链会带出 cut_region.py / path_generationSafe10.py，虽然U-Net算法本身不需要它们）。
#改成继承 cnn_recognizer_base.py 里的 BaseCharRecognizer——只有CNN加载和识别，不含任何
#切割域/路径搜索代码。这样U-Net这条路径在依赖上就是完全独立、干净的了，
#识别用的还是同一个训练好的CNN checkpoint（和图论方法保持一致，比较依然公平）。

import numpy as np
import torch

from cnn_recognizer_base import BaseCharRecognizer
from unet_seg_model import TinyUNet
from denoising1 import remove_salt_pepper_noise


class UNetLeftRightRecognizer(BaseCharRecognizer):
    def __init__(self, cnn_model_path, unet_weight_path, use_denoise=True):
        super().__init__(cnn_model_path)  # 只拿到CNN加载 + recognize_image/recognize_batch
        self.unet = TinyUNet().to(self.device)
        self.unet.load_state_dict(torch.load(unet_weight_path, map_location=self.device))
        self.unet.eval()
        self.use_denoise = use_denoise

    def _segment_with_unet(self, image_32x40_uint8):
        """输入去噪后(或原始)的 32x40 uint8 图，返回 left_img/right_img（非本字符像素置0），
        含义和图论方法里 generate_left_right_images 的输出完全一致。"""
        img_t = torch.from_numpy(image_32x40_uint8.astype(np.float32) / 255.0)
        img_t = img_t.unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_mask = self.unet(img_t).argmax(dim=1).squeeze(0).cpu().numpy()  # {0,1,2}
        left_img = np.where(pred_mask == 1, image_32x40_uint8, 0).astype(np.uint8)
        right_img = np.where(pred_mask == 2, image_32x40_uint8, 0).astype(np.uint8)
        return left_img, right_img, pred_mask

    def _preprocess(self, img):
        img_np = img.numpy().squeeze(axis=0)
        imgnp = (img_np * 255).astype('uint8')

        original_image = np.zeros((32, 40), dtype=np.uint8)
        h, w = imgnp.shape
        h_start = (32 - h) // 2
        w_start = (40 - w) // 2
        original_image[h_start:h_start + h, w_start:w_start + w] = imgnp

        if self.use_denoise:
            return remove_salt_pepper_noise(original_image, True)
        return original_image

    def process_single_image_unet(self, img, label):
        proc_image = self._preprocess(img)
        left_img, right_img, pred_mask = self._segment_with_unet(proc_image)

        left_pred, left_conf = self.recognize_image(left_img)
        right_pred, right_conf = self.recognize_image(right_img)

        if left_pred == 0:
            combined_pred = 100 + right_pred
        else:
            combined_pred = left_pred * 10 + right_pred

        is_correct = 1 if combined_pred == label else 0
        return [is_correct, left_conf, right_conf], [8], [combined_pred], label

    def process_batch_images_unet(self, img_data, num_images=None):
        actual_num = len(img_data) if num_images is None else min(num_images, len(img_data))
        rows = []
        correct = 0
        for i in range(actual_num):
            img, label = img_data[i]
            cut_results, path_id, path_lbl, true_label = self.process_single_image_unet(img, label)
            is_correct, left_conf, right_conf = cut_results
            combined_pred = path_lbl[0]
            correct += is_correct
            rows.append([i + 1, true_label, combined_pred, is_correct, left_conf, right_conf])

            if (i + 1) % 2000 == 0:
                print(f"已处理 {i+1}/{actual_num}，当前累计准确率 {correct/(i+1):.4f}")

        accuracy = correct / actual_num if actual_num else 0.0
        return rows, accuracy
