#unet_left_right_recog_cvl.py
#
#对应 u_left_right_recog1_cvl.py 的 U-Net 版本，用于CVL真实数据的 zero-shot 泛化测试：
#U-Net权重就是训练TDTS时得到的那个(unet_tdts.pt)，这里不做任何针对CVL的微调——
#因为CVL是真实手写数据，没有"哪个像素属于左/右字符"这种逐像素真值标注（只有两位数
#的整体标签），没法像TDTS那样监督训练分割掩码，所以这个脚本测的就是"U-Net只在合成
#数据上训练过，直接搬到真实数据上表现如何"，可以直接对照Table 11里图论方法的
#zero-shot(TDTS→CVL)跨域衰减幅度。
#
#和 u_left_right_recog1_cvl.py 的区别只在"分割"这一步：
#   原来：denoise -> compute_corridor -> find_connected_points -> generate_cut_lines
#         -> generate_left_right_images(按行cut_x切，再internally调standardize_for_recognition)
#   现在：denoise -> U-Net逐像素预测 -> 按类别置0得到left_img/right_img
#         -> 同样过 standardize_for_recognition (这一步对CVL识别准确率至关重要，
#            不能省略——识别模型是在标准化过的CVL Single Digit数据上训练的)
#
#识别模型继续用 cnn_recognizer_base_cvl.py 里同一个CVL CNN checkpoint，
#和图论方法保持完全一致，比较才公平。

import numpy as np
import torch

from cnn_recognizer_base_cvl import BaseCharRecognizerCVL
from unet_seg_model import TinyUNet
from denoising1 import remove_salt_pepper_noise
from recognition_preprocess_fix import standardize_for_recognition


class UNetLeftRightRecognizerCVL(BaseCharRecognizerCVL):
    def __init__(self, cnn_model_path, unet_weight_path, use_denoise=True):
        super().__init__(cnn_model_path)
        self.unet = TinyUNet().to(self.device)
        self.unet.load_state_dict(torch.load(unet_weight_path, map_location=self.device))
        self.unet.eval()
        self.use_denoise = use_denoise

    def _segment_with_unet(self, image_32x40_uint8):
        img_t = torch.from_numpy(image_32x40_uint8.astype(np.float32) / 255.0)
        img_t = img_t.unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_mask = self.unet(img_t).argmax(dim=1).squeeze(0).cpu().numpy()  # {0,1,2}
        left_img = np.where(pred_mask == 1, image_32x40_uint8, 0).astype(np.uint8)
        right_img = np.where(pred_mask == 2, image_32x40_uint8, 0).astype(np.uint8)
        return left_img, right_img, pred_mask

    def _preprocess(self, img):
        """和 u_left_right_recog1_cvl.py 的 process_single_image 前几步完全一致：
        PNG读进来的图（任意尺寸）居中放进32x40画布，再去噪。"""
        img_np = img.numpy().squeeze(axis=0)
        imgnp = (img_np * 255).astype('uint8')

        original_image = np.zeros((32, 40), dtype=np.uint8)
        h, w = imgnp.shape
        h_start = (32 - h) // 2
        w_start = (40 - w) // 2
        original_image[h_start:h_start + h, w_start:w_start + w] = imgnp

        if self.use_denoise:
            return remove_salt_pepper_noise(original_image)
        return original_image

    def process_single_image_unet(self, img, label):
        proc_image = self._preprocess(img)
        left_img, right_img, pred_mask = self._segment_with_unet(proc_image)

        # 关键一步：CVL识别模型是在标准化过的Single Digit数据上训练的，
        # 分割出来的半图必须先过这个函数，否则识别会因为输入分布不匹配而崩掉，
        # 那样测出来的就是"标准化没做对"而不是"U-Net分割好不好"。
        left_std = standardize_for_recognition(left_img)
        right_std = standardize_for_recognition(right_img)

        left_pred, left_conf = self.recognize_image(left_std)
        right_pred, right_conf = self.recognize_image(right_std)

        combined_pred = left_pred * 10 + right_pred  # 和CVL图论方法同一套编码，不做+100处理
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

            if (i + 1) % 200 == 0:
                print(f"已处理 {i+1}/{actual_num}，当前累计准确率 {correct/(i+1):.4f}")

        accuracy = correct / actual_num if actual_num else 0.0
        return rows, accuracy
