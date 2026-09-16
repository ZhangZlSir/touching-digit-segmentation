#u_left_right_recog.py
from itertools import repeat
import os
import struct
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from denoising1 import remove_salt_pepper_noise

import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from water12 import process_single_watershed

# 定义 CNN 模型（与训练时相同的结构）
class CNN(nn.Module):
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
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 10)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

class LeftRightRecognizer:
    def __init__(self, model_path='noise1_model_state_epoch_21.pth'):
        """初始化识别器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def _load_model(self, model_path):
        """加载预训练模型 - 使用状态字典方式"""
        # 创建模型实例
        model = CNN()
        # 加载状态字典
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def recognize_batch(self, image_list):
        """批量识别多张图片，返回预测数字列表和置信度列表"""
        if not image_list:
            return [], []
        
        # 批量预处理
        batch_tensors = []
        for image_array in image_list:
            image_pil = Image.fromarray(image_array.astype('uint8'))
            image_tensor = self.preprocess(image_pil)
            batch_tensors.append(image_tensor)
        
        # 堆叠成批次
        batch_tensor = torch.stack(batch_tensors)
        
        with torch.no_grad():
            output = self.model(batch_tensor)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            confidences = probs[torch.arange(probs.size(0)), preds]
        
        # 转换为Python列表
        pred_list = preds.tolist()
        confidence_list = confidences.tolist()
        
        return pred_list, confidence_list    
        

    def process_batch_images(self, img_data, num_images=100):
        """批量处理图片 - 真正的批量处理版本"""
        all_results = []
        
        # 确定实际处理的图片数量，防止索引越界
        actual_num_images = min(num_images, len(img_data))
        if actual_num_images == 0:
            print("警告: 没有可处理的图片数据")
            return all_results
        
        print(f"批量处理 {actual_num_images} 张图片...")
        
        # 第一阶段：收集所有图片数据
        print("收集图片数据...")
        original_images = []
        labels = []
        
        for image_index in range(actual_num_images):
            img, label = img_data[image_index]
            
            # 预处理图片
            img_np = img.numpy().squeeze(axis=0)
            imgnp = (img_np * 255).astype('uint8')
            
            # 创建32x40的图像
            original_image = np.zeros((32, 40), dtype=np.uint8)
            h, w = imgnp.shape
            h_start = (32 - h) // 2
            w_start = (40 - w) // 2
            original_image[h_start:h_start+h, w_start:w_start+w] = imgnp
            
            original_images.append(original_image)
            labels.append(label)
        
        # 第二阶段：批量去噪
        print("批量去噪...")
        denoised_images = []
        for img in original_images:
            #denoised_images.append(remove_salt_pepper_noise(img))
            denoised_images.append(img)  # 暂时不去噪，直接使用原图
        # with ProcessPoolExecutor(max_workers=4) as executor:
        #      # 批量提交任务，返回迭代器（按输入顺序返回结果）
        #     denoised_images = list(executor.map(remove_salt_pepper_noise, 
        #                                         original_images,
        #                                         repeat(True) # 第二个参数：给每张图传开关（循环复用）
        #                                         ))


        # 第三阶段：批量寻找连通点 分水岭直接在图像上找，不用切割域

        # 第五阶段：批量生成左右图像并收集（分水岭算法版本） 
        print("批量生成左右图像（分水岭算法）...")
        all_left_imgs = []  # 收集所有左侧图像
        all_right_imgs = [] # 收集所有右侧图像
        metadata = []  # 存储(图片索引, 路径索引, 真实标签, 路径ID)

        with ProcessPoolExecutor(max_workers=4) as executor:
            # 传递4个一一对应的参数列表，自动配对
            results = executor.map(
                process_single_watershed,  # 辅助函数
                denoised_images,           # 参数1：denoised_img
                labels,                    # 参数2：label
                range(actual_num_images)   # 参数3：image_index（原循环的i）
            )
            
            # 按顺序收集结果（和输入顺序完全一致，无错位）
            for left_img, right_img, meta in results:
                all_left_imgs.append(left_img)
                all_right_imgs.append(right_img)
                metadata.append(meta)

        # 第六阶段：批量识别 - 一次性识别所有左右图像
        print(f"批量识别: {len(all_left_imgs)} 张左侧图像和 {len(all_right_imgs)} 张右侧图像")
        
        # 检查是否有图像需要识别
        if not all_left_imgs or not all_right_imgs:
            print("警告: 没有生成的左右图像可供识别")
            return all_results
        
        # 批量识别左侧图像
        left_preds, left_confs = self.recognize_batch(all_left_imgs)
        
        # 批量识别右侧图像
        right_preds, right_confs = self.recognize_batch(all_right_imgs)
        
        # 第七阶段：组装结果
        print("组装结果...")
        # 按图片索引分组结果
        results_by_image = {}
        for i, meta in enumerate(metadata):
            image_index = meta['image_index']
            if image_index not in results_by_image:
                results_by_image[image_index] = []
            
            # 计算组合预测结果
            left_pred = left_preds[i]
            right_pred = right_preds[i]
            
            if left_pred == 0:
                combined_pred = 100 + right_pred
            else:
                combined_pred = left_pred * 10 + right_pred
            
            is_correct = 1 if combined_pred == meta['label'] else 0
            
            results_by_image[image_index].append({
                'path_id': meta['path_id'],
                'is_correct': is_correct,
                'left_conf': left_confs[i],
                'right_conf': right_confs[i],
                'combined_pred': combined_pred,
                'label': meta['label']
            })
        
        # 构建最终结果行
        for image_index, path_results in results_by_image.items():
            row = [image_index + 1, path_results[0]['label']]  # 图片ID和真实标签
            
            # 添加每个切割的结果
            for result in path_results:
                row.extend([
                    result['path_id'],           # 路径号
                    result['is_correct'],        # 是否正确
                    result['combined_pred'],     # 路径识别的标签
                    result['left_conf'],         # 左侧置信度
                    result['right_conf']         # 右侧置信度
                ])
            
            all_results.append(row)
        
        print(f"批量处理完成: 成功处理 {len(all_results)} 张图片")
        return all_results
  