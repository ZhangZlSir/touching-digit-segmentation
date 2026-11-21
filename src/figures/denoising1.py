#denoising1.py
import numpy as np
import torch
from numba import jit  # 新增：导入numba
# 11-21日修订，1 去除remove_salt_pepper_noise包含内部循环的代码，展开循环以提升性能
#   2 添加@jit装饰器以加速函数执行
# 去除椒盐噪声（保持不变）

def remove_salt_pepper_noise1(image):
    """基础版本 - 去除椒盐噪声"""
    denoised = image.clone()
    height, width = image.shape
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            current_pixel = image[y, x]
            
            neighborhood = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    neighborhood.append(image[y+i, x+j])
            
            neighborhood_mean = np.mean(neighborhood)
            neighborhood_std = np.std(neighborhood)
            
            if abs(current_pixel - neighborhood_mean) > 3 * neighborhood_std:
                denoised[y, x] = torch.tensor(np.median(neighborhood), dtype=denoised.dtype)
    
    return denoised

@jit(nopython=True, cache=True)  # cache=True：
def remove_salt_pepper_noise(image,enable_denoise=True):
    """优化版本 - 保持算法逻辑不变"""
    if not enable_denoise:
        return image  # 不开启去噪，直接返回原图
    denoised = image.copy() ## 注意：这里使用的是numpy数组的copy方法，不是torch的clone方法
    height, width = image.shape
    # 预分配内存
    neighborhood = np.zeros(8, dtype=image.dtype) #经测试，新的python版本已经修复了np.median可能溢出的bug
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            # 手动展开循环，避免嵌套
            neighborhood[0] = image[y-1, x-1]
            neighborhood[1] = image[y-1, x]
            neighborhood[2] = image[y-1, x+1]
            neighborhood[3] = image[y, x-1]
            neighborhood[4] = image[y, x+1]
            neighborhood[5] = image[y+1, x-1]
            neighborhood[6] = image[y+1, x]
            neighborhood[7] = image[y+1, x+1]
            
            current_pixel = image[y, x]
            mean_val = np.mean(neighborhood)
            std_val = np.std(neighborhood)
            
            if abs(current_pixel - mean_val) > 3 * std_val:
                denoised[y, x] = np.median(neighborhood)
    
    return denoised