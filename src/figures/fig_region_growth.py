#fig_region_growth.py  --  论文 Fig 7：切割域生长的 14 步过程（4.5 目录最终版）
# 主程序：展示14步区域生长过程
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
from PIL import Image
from matplotlib.gridspec import GridSpec
import matplotlib

# 设置符合期刊要求的字体
# 期刊允许：Arial, Times, 或 Symbol
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Times New Roman']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # 数学字体

# 导入自定义模块
from denoising1 import remove_salt_pepper_noise
from region_up import trace_region_growth_with_steps

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = str(PROJECT_ROOT / 'data' / 'tdts_dataset')
OUT_DIR = PROJECT_ROOT / 'results' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============== 配置常量 ==============
# 图像尺寸常量
IMAGE_HEIGHT = 32
FULL_WIDTH = 40

# 裁剪配置（17宽方案：12-28区域）
CROP_X_START = 12
CROP_X_END = 28
CROP_WIDTH = CROP_X_END - CROP_X_START + 1  # 17

# 期刊图片要求
JOURNAL_DPI = 300  # 300-600 dpi 范围内
MAX_WIDTH_PIXELS = 2250  # 最大宽度（300 dpi时）
MIN_WIDTH_PIXELS = 789   # 最小宽度（300 dpi时）
MAX_HEIGHT_PIXELS = 2625 # 最大高度（300 dpi时）

# ============== 工具函数 ==============
def crop_to_region(image):
    """
    将图像裁剪到12-28区域（17宽），只裁剪不缩放
    Args:
        image: 输入图像，32×40 numpy数组
    Returns:
        cropped_image: 裁剪后的图像（17×32）
    """
    # 确保输入是numpy数组
    if hasattr(image, 'numpy'):
        image_np = image.numpy().squeeze()
    else:
        image_np = image.copy()
    
    # 裁剪到指定区域
    if len(image_np.shape) == 2:
        return image_np[:, CROP_X_START:CROP_X_END+1]
    elif len(image_np.shape) == 3:
        return image_np[:, CROP_X_START:CROP_X_END+1, :]
    else:
        return image_np

def mark_points_with_color_cycle(image, points_list):
    """
    生成最终图像，按照新的颜色规则：
    当前步骤新增的点 = 红色
    历史步骤已存在的点 = 蓝色
    
    Args:
        image: 输入图像 (32×40)
        points_list: 点列表的列表，每个元素是一个步骤的点集
    Returns:
        colored_image: 标记后的彩色图像 (32×40×3)
    """
    # 创建基础彩色图像
    if len(image.shape) == 2:
        base_image = np.stack([image] * 3, axis=2)
    else:
        base_image = image.copy()
    
    # 确保是3通道
    if base_image.shape[2] != 3:
        base_image = np.repeat(base_image, 3, axis=2)
    
    # 确保points_list有至少一步
    if not points_list:
        return base_image
    
    # 获取最后一步的点集
    final_points = set(points_list[-1])
    
    # 如果没有历史步骤（只有第一步）
    if len(points_list) == 1:
        # 所有点都是当前步骤新增的点，标记为红色
        for point in points_list[0]:
            x, y = point
            if 0 <= x < base_image.shape[1] and 0 <= y < base_image.shape[0]:
                base_image[y, x, 0] = 255  # 红色
                base_image[y, x, 1] = 0
                base_image[y, x, 2] = 0
        return base_image
    
    # 获取上一步的点集
    previous_points = set(points_list[-2])
    
    # 计算当前步骤新增的点 = 最后一步 - 上一步
    new_points = final_points - previous_points
    historical_points = previous_points
    
    # 标记历史点（蓝色）
    for point in historical_points:
        x, y = point
        if 0 <= x < base_image.shape[1] and 0 <= y < base_image.shape[0]:
            base_image[y, x, 0] = 0  # 蓝色
            base_image[y, x, 1] = 0
            base_image[y, x, 2] = 255
    
    # 标记当前步骤新增的点（红色）
    for point in new_points:
        x, y = point
        if 0 <= x < base_image.shape[1] and 0 <= y < base_image.shape[0]:
            base_image[y, x, 0] = 255  # 红色
            base_image[y, x, 1] = 0
            base_image[y, x, 2] = 0
    
    return base_image

def up_region_14steps(img):
    """
    对单张图像执行14步区域生长
    返回：裁剪后的14个步骤图像列表和原始图像
    """
    # 确保图像是numpy数组
    if hasattr(img, 'numpy'):
        img_np = (img.numpy().squeeze() * 255).astype(np.uint8)
    else:
        img_np = img.astype(np.uint8)
    
    # 执行14步区域生长
    imgs, points = trace_region_growth_with_steps(img_np, 100)
    
    if not imgs:
        print("警告：区域生长未找到起点")
        return [], img_np
    
    # 为每个步骤创建彩色标记图像
    colored_images = []
    for step_idx, (img_step, points_step) in enumerate(zip(imgs, points)):
        # 为当前步骤创建彩色标记图像
        colored_img = mark_points_with_color_cycle(img_step, points[:step_idx+1])
        
        # 裁剪到12-28区域
        cropped_img = crop_to_region(colored_img)
        colored_images.append(cropped_img)
        
        # 如果已经收集了14张图像，提前退出
        if len(colored_images) >= 14:
            break
    
    # 如果步骤不足14步，用最后一步补齐
    while len(colored_images) < 14:
        colored_images.append(colored_images[-1] if colored_images else 
                            np.zeros((IMAGE_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8))
    
    return colored_images, img_np

def display_hybrid_growth_2row(original_image, growth_images, step_labels=None, save_path=None):
    """
    显示2行混合布局：总共14个步骤
    第一行：左侧40宽原图 + 右侧6个生长图像（步骤1-6）
    第二行：左侧图例 + 右侧7个生长图像（步骤7-13）
    
    修改：调整尺寸和保存格式以满足期刊要求
    """
    # 定义常量
    IMAGE_HEIGHT = 32
    CROP_WIDTH = 17
    CROP_X_START = 13
    ORIGINAL_WIDTH = 40
    LEGEND_WIDTH = 20
    SPACING = 3
    
    # 确保有足够的生长图像（至少13步）
    n_steps = len(growth_images)
    if n_steps < 13:
        print(f"警告：期望至少13步，但只得到{n_steps}步")
        # 补齐到13步
        last_image = growth_images[-1] if growth_images else np.zeros((IMAGE_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)
        growth_images = growth_images + [last_image] * (13 - n_steps)
    
    # 计算每行宽度
    row1_width = ORIGINAL_WIDTH + SPACING + 6 * CROP_WIDTH + 5 * SPACING
    row2_width = LEGEND_WIDTH + SPACING + 7 * CROP_WIDTH + 6 * SPACING
    
    # 根据期刊要求调整图形尺寸
    # 计算以英寸为单位的尺寸（300 dpi）
    fig_width_inches = 7.5  # 2250 pixels / 300 dpi = 7.5 inches
    fig_height_inches = 4.0  # 1200 pixels / 300 dpi = 4.0 inches
    
    # 创建图形 - 调整尺寸以满足期刊要求
    fig = plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=JOURNAL_DPI)
    
    # ========== 第一行：原图 + 步骤1-6 ==========
    # 调整子图位置以适应新尺寸
    gs_row1 = GridSpec(1, 1, figure=fig, bottom=0.55, top=0.95, left=0.05, right=0.95, hspace=0)
    ax_row1 = fig.add_subplot(gs_row1[0])
    
    # 创建第一行画布
    row1_canvas = np.ones((IMAGE_HEIGHT, row1_width, 3), dtype=np.uint8) * 255
    
    # 放置原图
    if len(original_image.shape) == 2:
        original_rgb = np.stack([original_image] * 3, axis=2)
    else:
        original_rgb = original_image
    
    # 调整原图大小
    if original_rgb.shape[1] != ORIGINAL_WIDTH:
        original_pil = Image.fromarray(original_rgb.astype(np.uint8))
        original_pil = original_pil.resize((ORIGINAL_WIDTH, IMAGE_HEIGHT))
        original_rgb = np.array(original_pil)
    
    row1_canvas[:, :ORIGINAL_WIDTH, :] = original_rgb
    
    # 在原图上标记裁剪区域
    for i in range(0, IMAGE_HEIGHT, 4):
        if i < IMAGE_HEIGHT:
            row1_canvas[i, CROP_X_START, :] = [255, 0, 0]
            row1_canvas[i, CROP_X_START+CROP_WIDTH-1, :] = [255, 0, 0]
            if i+1 < IMAGE_HEIGHT:
                row1_canvas[i+1, CROP_X_START, :] = [255, 0, 0]
                row1_canvas[i+1, CROP_X_START+CROP_WIDTH-1, :] = [255, 0, 0]
    
    # 放置前6个生长图像 (步骤1-6)
    current_x = ORIGINAL_WIDTH + SPACING
    
    for i in range(6):
        if i < len(growth_images):
            img = growth_images[i]
            if len(img.shape) == 2:
                img_rgb = np.stack([img] * 3, axis=2)
            else:
                img_rgb = img
            
            row1_canvas[:, current_x:current_x+CROP_WIDTH, :] = img_rgb
            current_x += CROP_WIDTH + SPACING
    
    ax_row1.imshow(row1_canvas)
    ax_row1.set_xticks([])
    ax_row1.set_yticks([])
    
    # 在第一行上方添加标题 - 调整字体大小符合期刊要求
    row1_titles = ["Original"] + [f"Step {i+1}" for i in range(6)]
    row1_centers = []
    
    # 原图中心
    row1_centers.append(ORIGINAL_WIDTH/2 / row1_width)
    # 步骤1-6中心
    for i in range(6):
        start_x = ORIGINAL_WIDTH + SPACING + i * (CROP_WIDTH + SPACING)
        center_x = start_x + CROP_WIDTH/2
        row1_centers.append(center_x / row1_width)
    
    # 使用符合期刊要求的字体大小（8-12 point）
    title_fontsize = 10  # 在8-12点范围内
    
    for center_x_ratio, title in zip(row1_centers, row1_titles):
        fig_x = 0.05 + (0.95 - 0.05) * center_x_ratio
        fig.text(fig_x, 0.96, title, fontsize=title_fontsize, 
                horizontalalignment='center', verticalalignment='top')#fontweight='bold'
    
    # ========== 第二行：图例 + 步骤7-13 ==========
    # 第二行分成8部分：图例(1/8) + 7个步骤(各1/8)
    gs_row2 = GridSpec(1, 8, figure=fig, bottom=0.05, top=0.45, 
                       left=0.05, right=0.95, wspace=0.05, hspace=0)
    
    # 第1部分：图例
    ax_legend = fig.add_subplot(gs_row2[0])
    ax_legend.clear()
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    
    # 绘制图例
    ax_legend.set_facecolor('white')
    ax_legend.text(0.5, 0.95, "Legend", fontsize=12, 
                   transform=ax_legend.transAxes,
                   horizontalalignment='center', verticalalignment='top')
    
    # 图例项目
    legend_items = [
        ((1.0, 0, 0), "New"),
        ((0, 0, 1.0), "History")
    ]
    
    y_pos = 0.7
    y_step = 0.25
    
    for color, label in legend_items:
        # 颜色方块
        rect = plt.Rectangle((0.1, y_pos-0.1), 0.25, 0.15, 
                             facecolor=color, transform=ax_legend.transAxes)
        ax_legend.add_patch(rect)
        
        # 文本标签 - 使用符合期刊要求的字体大小
        ax_legend.text(0.45, y_pos-0.025, label, fontsize=10,
                       transform=ax_legend.transAxes, verticalalignment='center')
        y_pos -= y_step
    
    # 第2-8部分：步骤7-13 (对应生长图像索引6-12)
    row2_titles = [f"Step {i+7}" for i in range(7)]
    
    for i in range(7):
        ax_img = fig.add_subplot(gs_row2[i+1])
        img_idx = 6 + i  # 步骤7-13对应索引6-12
        
        if img_idx < len(growth_images):
            img = growth_images[img_idx]
            
            if len(img.shape) == 2:
                ax_img.imshow(img, cmap='gray')
            else:
                ax_img.imshow(img)
            
            # 标题 - 使用符合期刊要求的字体大小
            ax_img.set_title(row2_titles[i], fontsize=10)
            
            ax_img.set_xticks([])
            ax_img.set_yticks([])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存图片 - 按照期刊要求修改
    if save_path:
        # 确保使用TIFF格式
        if not save_path.lower().endswith(('.tif', '.tiff')):
            save_path = save_path.rsplit('.', 1)[0] + '.tiff'
        
        # 按照期刊要求保存TIFF文件
        plt.savefig(save_path, 
                   dpi=JOURNAL_DPI,  # 300 dpi，在300-600范围内
                   bbox_inches='tight', 
                   pad_inches=0.1,  # 减少多余空白
                   facecolor='white', 
                   edgecolor='none',
                   format='tiff',  # 明确指定TIFF格式
                   pil_kwargs={'compression': 'tiff_lzw'})
        plt.savefig(OUT_DIR / "Fig1_region_growth_14steps.pdf", dpi=JOURNAL_DPI, bbox_inches='tight',
                   pad_inches=0.1,
                   facecolor='white', edgecolor='none',
                   format='pdf')
        
        print(f"✓ 混合布局图片已保存为TIFF格式: {save_path}")
        print(f"✓ 图片尺寸: {fig_width_inches:.1f}×{fig_height_inches:.1f} 英寸")
        print(f"✓ 分辨率: {JOURNAL_DPI} dpi")
        print(f"✓ 像素尺寸: {int(fig_width_inches*JOURNAL_DPI)}×{int(fig_height_inches*JOURNAL_DPI)} 像素")
    
    plt.show()
    return fig

# ============== 主程序 ==============
def main():
    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor()])
    img_data = MNIST(DATA_ROOT, download=False, transform=transform, train=True)
    
    # 测试样本（使用第9张图像，索引3082）
    test_index = 3082
    img, label = img_data[test_index]
    
    print(f"处理图像索引: {test_index}, 标签: {label}")
    
    # 去噪处理
    img_np = (img.numpy().squeeze() * 255).astype(np.uint8)
    no_noise_img = remove_salt_pepper_noise(img_np)
    
    # 执行14步区域生长
    growth_images, original_img = up_region_14steps(no_noise_img)
    
    if growth_images:
        print(f"✓ 成功生成 {len(growth_images)} 步区域生长图像")
        print(f"✓ 调整图像格式以满足PLOS期刊要求")
        print(f"  - 格式: TIFF (LZW压缩)")
        print(f"  - 分辨率: {JOURNAL_DPI} dpi")
        print(f"  - 宽度范围: {MIN_WIDTH_PIXELS}-{MAX_WIDTH_PIXELS} 像素")
        print(f"  - 字体: Arial/Times, 8-12 point")
        
        # 显示混合布局
        fig = display_hybrid_growth_2row(
            original_img, 
            growth_images, 
            step_labels=[str(i) for i in range(13)], 
            save_path=str(OUT_DIR / 'Fig1_region_growth_14steps.tiff')  # 按期刊建议命名
        )
        
        print(f"\n✓ 区域生长展示完成")
        print(f"✓ 图像已保存为: Fig1_region_growth_14steps.tiff")
        print(f"✓ 符合PLOS One期刊图片格式要求")
    else:
        print("✗ 区域生长失败")

if __name__ == "__main__":
    main()