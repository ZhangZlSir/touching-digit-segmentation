#unet_recenter_utils.py
#
#"方案B"：不改U-Net、不重新训练，用corridor_utils.compute_corridor算出的走廊中心，
#把整张图水平平移对齐到TDTS训练时隐含的固定中心(x=20)，再喂给已经训练好的U-Net。
#
#背景：TDTS虽然没有显式调用compute_corridor，但它的走廊固定在x∈[13,27]，中心恒为20——
#本质上等价于"假设两个字符加起来的内容，永远大致居中在x=20附近"这个先验。U-Net在这种
#位置分布单一的数据上训练，很可能隐式学到了"该在x=20附近找边界"这个捷径。CVL图像是把
#提取出来的内容整体居中贴进32x40画布(容忍不同宽度w，居中偏移量(40-w)//2随w变化)，
#内容的实际水平位置分布跟TDTS完全不同，所以这里用同一套走廊计算逻辑重新对齐，
#看看能不能把这部分"绝对位置错位"的影响先剥离出来，跟"真实笔迹视觉风格差异"的
#影响分开看。

import numpy as np
from corridor_utils import compute_corridor


def recenter_by_corridor(image_32x40, margin_frac=0.15, target_center=20.0, fallback_center=20.0):
    """
    用compute_corridor算出的中心，把图像水平平移对齐到target_center。
    margin_frac默认0.15，和你CVL草稿5.X节里写的一致；如果你实际代码里用的是
    0.25或0.1，按你实际用的值改这里，保证和图论方法那边算走廊时用的margin_frac一致，
    这样"对齐目标"和"图论方法实际用的走廊"才是同一套逻辑，比较才有意义。

    返回：(recentered_image, shift_used, corridor_dict_or_None)
    """
    corridor = compute_corridor(image_32x40, margin_frac=margin_frac)
    if corridor is None:
        # 算不出走廊（比如整张图没有前景内容），不平移，原样返回
        return image_32x40.copy(), 0, None

    shift = int(round(target_center - corridor['center']))
    h, w = image_32x40.shape
    recentered = np.zeros_like(image_32x40)

    if shift == 0:
        recentered = image_32x40.copy()
    elif shift > 0:
        # 内容整体右移shift像素（相当于原图左边留白，右边被裁掉shift列）
        recentered[:, shift:] = image_32x40[:, :max(0, w - shift)]
    else:
        s = -shift
        recentered[:, :max(0, w - s)] = image_32x40[:, s:]

    return recentered, shift, corridor
