#u_left_right_recog1.py
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
from cut_region import find_connected_points
from path_generationSafe11 import generate_cut_lines
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from itertools import repeat
from corridor_utils import compute_corridor
from recognition_preprocess_fix import standardize_for_recognition

class CNN(nn.Module):
    """
    与 cnn4032-cvl-improved.py 中完全相同的结构：
    - 3层卷积（带BN）+ 2次MaxPool
    - 双Dropout（卷积后和全连接后）
    - 输入尺寸 1x32x40
    """
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
        self.dropout1 = nn.Dropout(dropout_p * 0.4)   # 卷积输出后的轻量 dropout
        self.dropout2 = nn.Dropout(dropout_p)         # 全连接后的 dropout
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

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_model = PROJECT_ROOT / 'models' / 'cvl_cnn_improved.pth'
class LeftRightRecognizer:
    def __init__(self, model_path=DATA_model):
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
    
    def recognize_image(self, image_array):
        """识别单张图片，返回预测数字和置信度"""
        image_pil = Image.fromarray(image_array.astype('uint8'))
        image_tensor = self.preprocess(image_pil).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item()
        
        return pred, confidence
    
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

    from recognition_preprocess_fix import standardize_for_recognition

    def generate_left_right_images(self, denoised_image, cut_lines):
        height, width = denoised_image.shape
        left_images = []
        right_images = []
        
        for i, path in enumerate(cut_lines):
            left_img = np.zeros_like(denoised_image)
            right_img = np.zeros_like(denoised_image)
            
            last_cut_x = width // 2
            for y in range(height):
                path_points_in_row = [point for point in path if point[1] == y]
                if path_points_in_row:
                    cut_x = min(point[0] for point in path_points_in_row)
                    last_cut_x = cut_x
                else:
                    cut_x = last_cut_x  # 沿用最近一次的切割位置，不再整行复制给两边
                left_img[y, :cut_x+1] = denoised_image[y, :cut_x+1]
                right_img[y, cut_x:] = denoised_image[y, cut_x:]
            
            left_images.append(standardize_for_recognition(left_img))
            right_images.append(standardize_for_recognition(right_img))
        
        return left_images, right_images
    
    def process_single_image(self, img, label):
        """处理单张图片，返回切割数量和识别结果"""
        # 预处理图片
        img_np = img.numpy().squeeze(axis=0)
        imgnp = (img_np * 255).astype('uint8')
        
        # 创建32x40的图像
        original_image = np.zeros((32, 40), dtype=np.uint8)
        h, w = imgnp.shape
        h_start = (32 - h) // 2
        w_start = (40 - w) // 2
        original_image[h_start:h_start+h, w_start:w_start+w] = imgnp
        
        # 去噪
        denoised_image = remove_salt_pepper_noise(original_image)
        #denoised_image = original_image  # 暂时不去噪，直接使用原图

        # 寻找连通点
        corridor = compute_corridor(denoised_image, margin_frac=0.25)
        x_range = (corridor['corridor_left'], corridor['corridor_right']) if corridor else (13, 27)
        points = find_connected_points(denoised_image, x_range=x_range)
        
        # 创建flag
        flag = np.zeros((32, 40), dtype=np.uint8)
        for x, y in points:
            flag[y, x] = 1
        
        # 生成切割线
        if corridor:
            dynamic_center = corridor['center']   # 几何与质量的平均值
        else:
            dynamic_center = 20
        # 生成切割线
        cut_lines = generate_cut_lines(flag, denoised_image, max_paths=7, center_x=dynamic_center)
        #cut_lines = generate_cut_lines(flag, denoised_image, max_paths=7)
        path_points_list = [path_info["path"] for path_info in cut_lines]
        # 生成左右图片
        left_images, right_images = self.generate_left_right_images(denoised_image, path_points_list)
        
        # 识别结果
        results = []
        identifiers = []
        vote_values = []
        
        for i, (left_img, right_img) in enumerate(zip(left_images, right_images)):
            path_info = cut_lines[i]
            identifier = path_info["identifier"]
            #vote_value = path_info["vote_value"] #修改为实际预测值，不再使用投票值
            identifiers.append(identifier)
            #vote_values.append(vote_value) #修改为实际预测值，不再使用
            # 识别左侧图片
            left_pred, left_conf = self.recognize_image(left_img)
            
            # 识别右侧图片  
            right_pred, right_conf = self.recognize_image(right_img)
            
            # 组合预测结果
            combined_pred = left_pred * 10 + right_pred

            vote_value = combined_pred
            vote_values.append(vote_value)

            # 检查是否正确
            is_correct = 1 if combined_pred == label else 0
            
            results.extend([is_correct, left_conf, right_conf])
        
        return results, identifiers, vote_values, label

    def process_batch_images1(self, img_data, num_images=400):#此函数不对？
        """批量处理图片"""
        all_results = []
        
        for image_index in tqdm(range(num_images), desc="Processing Images"):
            img, label = img_data[image_index]
            cut_results, path_id,path_vote,true_label = self.process_single_image(img, label)
            
            # 构建结果行
            row = [image_index + 1, true_label]
            
            # 添加每个切割的结果
            for i in range(len(path_id)):
                start_idx = i * 3
                row.extend([
                    cut_results[start_idx],     # 是否正确
                    cut_results[start_idx + 1], # 左侧置信度
                    cut_results[start_idx + 2]  # 右侧置信度
                ])
            
            all_results.append(row)
        
        return all_results

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
        # for img in original_images:
        #     #denoised_images.append(remove_salt_pepper_noise(img))
        #     denoised_images.append(img)  # 暂时不去噪，直接使用原图
        with ProcessPoolExecutor(max_workers=4) as executor:
        # 批量提交任务，返回迭代器（按输入顺序返回结果）
            denoised_images = list(executor.map(remove_salt_pepper_noise, 
                                            original_images,
                                            repeat(True) # 第二个参数：给每张图传开关（循环复用）
                                            ))
        # 第三阶段：批量寻找连通点
        print("批量寻找连通点...")
        all_points = []
        # for img in denoised_images:
        #     all_points.append(find_connected_points(img))
        corridors = [compute_corridor(img, margin_frac=0.1) for img in denoised_images]
        x_ranges = [(c['corridor_left'], c['corridor_right']) if c else (13, 27) for c in corridors]
        with ProcessPoolExecutor(max_workers=4) as executor:
            all_points = list(executor.map(find_connected_points, denoised_images, x_ranges))
        
        # 第四阶段：批量生成切割线
        print("批量生成切割线...")
        all_cut_lines = []
        for i, (points, denoised_img) in enumerate(zip(all_points, denoised_images)):
            flag = np.zeros((32, 40), dtype=np.uint8)
            
            for x, y in points:
                flag[y, x] = 1
             # ---- 从 corridors 中提取动态中心 ----
            if corridors and i < len(corridors) and corridors[i] is not None:
                # 使用 corridor['center']（几何与质量中心的平均）
                center_x = int(round(corridors[i]['center']))
                # 可选：也可以使用 corridor_left/right 来进一步约束，但仅 center 已足够
            else:
                center_x = 20  # 保底值
            cut_lines = generate_cut_lines(flag, denoised_img, max_paths=7, center_x=center_x)
            all_cut_lines.append(cut_lines)
        
        # 第五阶段：批量生成左右图像并收集
        print("批量生成左右图像...")
        all_left_imgs = []  # 收集所有左侧图像
        all_right_imgs = [] # 收集所有右侧图像
        metadata = []  # 存储(图片索引, 路径索引, 真实标签, 路径ID)
        
        for i, (cut_lines, denoised_img, label) in enumerate(zip(all_cut_lines, denoised_images, labels)):
            path_points_list = [path_info["path"] for path_info in cut_lines]
            left_images, right_images = self.generate_left_right_images(denoised_img, path_points_list)
            
            for path_idx, (left_img, right_img) in enumerate(zip(left_images, right_images)):
                all_left_imgs.append(left_img)
                all_right_imgs.append(right_img)
                path_info = cut_lines[path_idx]
                metadata.append({
                    'image_index': i,
                    'path_idx': path_idx,
                    'label': label,
                    'path_id': path_info["identifier"]
                })
        
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

    def save_results_to_csv(self, all_results, output_file='cutting_recognition_results.csv'):
        """保存结果到CSV文件"""
        # 确定最大切割数
        max_cuts = max(row[1] for row in all_results)
        
        # 创建列名
        columns = ['imgID', 'PathNum', 'TrueLbl']
        for i in range(max_cuts):
            columns.extend([
                f'P{i+1}OK',
                f'P{i+1}LC', 
                f'P{i+1}RC'
            ])
        
        # 创建DataFrame
        df = pd.DataFrame(all_results, columns=columns)
        
        # 保存到CSV
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"结果已保存到 {output_file}")
        
        # 计算并打印统计信息
        self._calculate_statistics(all_results)
    
    def _calculate_statistics(self, all_results):
        """计算统计信息"""
        total_correct = 0
        total_predictions = 0
        
        for row in all_results:
            num_cuts = row[1]
            for i in range(num_cuts):
                if row[3 + i * 3] == 1:  # 检查是否正确
                    total_correct += 1
                total_predictions += 1
        
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        print(f"总体识别准确率: {accuracy:.4f} ({total_correct}/{total_predictions})")