# 连通手写数字的图论分割方法与数据集构建

本仓库是论文 **"Research on graph theory segmentation method and dataset construction for connected handwritten digits"**（PLOS ONE，在审）的完整开源代码与数据，包含：

- **TDTS 数据集**（Two-digit touching segmentation，两位数字粘连分割）的构建流程与生成代码；
- 本文提出的 **切割域（cutting domain）图论分割框架** 及 **A\* / Dijkstra** 系列算法的实现；
- 对比方法 **VPM（垂直投影）**、**WA（分水岭）** 的实现；
- **U-Net 分割对比实验**；
- **CVL 真实手写数字串** 上的零样本泛化评估。

---

## 论文信息

- **标题**：Research on graph theory segmentation method and dataset construction for connected handwritten digits
- **作者**：Zhilai Zhang, Yang Zhang, Yang Wang
- **通讯作者**：zhangzl@hdc.edu.cn
- **单位**：邯郸学院信息工程学院，河北科技大学信息科学与工程学院
- **期刊**：PLOS ONE（在审）
- **核心贡献**：
  1. 构建标准化、类别均衡的 **TDTS 双字符粘连数据集**，覆盖 “00”–“99” 全部 100 类；
  2. 提出 **切割域（cutting domain）图论分割框架**，将分割问题解耦为“结构化搜索空间构建 + 图论最优路径搜索”；
  3. 系统验证 **Dijkstra** 与 **A\* 系列算法**（Left-Biased、Right-Biased、Unbiased）的性能，并与 VPM、WA、U-Net 对比；
  4. 在 **CVL 真实手写数字串** 上验证零样本泛化能力。
- **关键结果**：
  - TDTS-TS 上 **Left-Biased A\*** 识别准确率 **93.13%**；
  - CVL 上 **Unbiased A\*** 保持 **91.4%** 准确率；
  - 监督 U-Net 在跨域时从 **95.90%** 降至 **53.05%**，而无监督图论方法仅下降 **1.28** 个百分点。

---

## 一、目录总览

```text
touching-digit-segmentation/
├── data/        数据集（构建输入 + 产出 + CVL 基准）
├── models/      训练好的模型权重
├── results/     实验输出 CSV
├── src/         全部源代码
└── README.md    本文件
```

---

## 二、`data/` 数据目录

```text
data/
├── README.md                   数据说明（重要，先读这份）
├── mnist_compact_20x20/        MNIST 20×20 压缩字符（TDTS 构建原料）
├── tdts_build_inputs/          TDTS 构建输入（坐标 + 字符对索引）
├── tdts_dataset/               TDTS 双字符粘连库（论文核心数据集）
├── tdts_masks/                 U-Net 像素级真值掩码
├── tdts_single_char/           三种单字符识别模型训练集
│   ├── clean/                  → CRM（不带噪声）
│   ├── noisy/                  → NRM（带噪声）
│   └── mixed/                  → MRM（混合）
├── cvl_single_char.zip         CVL 单字符集（解压后用于训练 CVL CNN）
└── cvl_double_char.zip         CVL 双字符粘连集（论文 Table 11 用）
```

**说明**

| 目录 / 文件 | 内容 | 论文对应 |
|---|---|---|
| `mnist_compact_20x20/` | 从 MNIST 裁剪出的 20×20 紧凑字符，TDTS 构建的基本单元 | 第 3.3.1 节 |
| `tdts_build_inputs/` | 4 个 CSV：右字符放置坐标、左右字符对索引（train / test 各一份） | 第 3.2 / 3.3 节 |
| `tdts_dataset/` | TDTS 图像集，60,000 训练 + 10,000 测试，MNIST idx-ubyte 格式 | 第 3.4 节；Table 1、2；Table 4/8/9 为基于该数据集的实验 |
| `tdts_masks/` | U-Net 训练用的逐像素掩码（0=背景、1=左字符、2=右字符） | 第 4.4 节；第 5.5 节 / Table 12 |
| `tdts_single_char/` | 三种识别模型的训练集（CRM / NRM / MRM） | 第 5.2.1 节；Table 3 |
| `cvl_single_char.zip` | CVL 单字符集，用于训练 CVL CNN。训练集 7,000，测试集 21,780，单字符测试准确率 97.06% | 第 5.4.1 节；Table 10 |
| `cvl_double_char.zip` | CVL 双字符粘连集，共 1,953 个样本，覆盖 75 类，用于零样本泛化评估 | 第 5.4.2 节；Table 11 |

**压缩包解压方法**（Windows / Linux 均可，假设当前位于项目根目录）：

```bash
unzip data/cvl_single_char.zip
unzip data/cvl_double_char.zip
```

解压后目录结构为：

```text
data/
├── cvl_single_char/
│   ├── trainStd40-32/
│   └── evalStd40-32/
└── cvl_double_char/        # 直接包含 PNG
```

---

## 三、`src/` 源代码目录

```text
src/
├── common/                 通用工具
├── recognizer_training/    单字符识别模型训练
├── tdts_build/             TDTS 数据集构建
├── tdts_pipelines/         TDTS 上的图论分割方法（本文方法 + VPM + WA）
├── cvl_pipelines/          CVL 上的图论分割方法
├── tdts_unet/              TDTS 上的 U-Net 分割对比
└── cvl_unet/               CVL 上的 U-Net 零样本对比
```

### 3.1 `src/common/` — 通用工具

- `dataset_viewer.py`：快速查看 TDTS 数据集的样本图，方便验证数据是否加载正确。

### 3.2 `src/recognizer_training/` — 单字符识别模型训练

- `train_tdts_recognizer.py`：训练 CRM / NRM / MRM。
- `train_cvl_recognizer.py`：训练 CVL 识别 CNN（含增强版）。

### 3.3 `src/tdts_build/` — TDTS 构建

- `B-new3240-noise2img.py`：从 MNIST 压缩字符 + 坐标 CSV 生成 TDTS 双字符粘连图。

### 3.4 `src/tdts_pipelines/` — TDTS 上的分割实验

- `cut_regionTest/`：本文提出的切割域 + 路径搜索（含消融实验开关）。
- `cut_line_all/`：切割线同时呈现图。
- `water_all/`：分水岭算法（WA）。
- `deNoise_water/`：分水岭 + 去噪。

每个子目录内均自带 `denoising1.py`、`path_generationSafe10.py` 等依赖，保证可以脱离其他目录独立运行。

### 3.5 `src/cvl_pipelines/` — CVL 上的分割实验

- `batch_main_recog2.py`：批量评估主入口（输出到 `outPutCVlTest/`）。
- `corridor_utils.py`：CVL 走廊边界与中心计算。
- `cut_region.py`：切割域构建。
- `path_generationSafe11.py`：路径搜索（A\* / Dijkstra 系列）。
- `u_left_right_recog1_cvl.py`：识别流水线主类。
- `recognition_preprocess_fix.py`：识别前处理（归一化）。
- `denoising1.py`：去噪。
- `tool/`：辅助脚本（CSV 合并、结果统计）。

### 3.6 `src/tdts_unet/` — TDTS 上的 U-Net 对比

- `mask3240_gen.py`：生成 U-Net 训练用的逐像素掩码。
- `unet_seg_model.py`：TinyUNet 网络定义。
- `unet_seg_train.py`：训练 U-Net。
- `batch_main_unet_recog.py`：批量评估 U-Net + MRM。
- `unet_left_right_recog.py`：U-Net 分割 + 识别流水线。

### 3.7 `src/cvl_unet/` — CVL 上的 U-Net 零样本对比

- `batch_main_unet_recog_cvl.py`：CVL 上 U-Net 零样本评估。
- `cnn_recognizer_base_cvl.py`：CVL CNN 加载器。
- `unet_left_right_recog_cvl.py`：U-Net 分割 + CVL CNN 识别。
- `unet_recenter_utils.py`：CVL 图像重定位到 32×40 输入。
- `unet_cvl_visual_qc_tool.py`：可视化质量控制工具。

---

## 四、`models/` 模型权重

```text
models/
├── README.md
├── tdts_crm.pth            CRM 识别模型（Clean）
├── tdts_nrm.pth            NRM 识别模型（Noisy）
├── tdts_mrm.pth            MRM 识别模型（Mixed，论文主实验用）
├── cvl_cnn_improved.pth    CVL 识别模型（增强版）
└── unet_tdts.pt            U-Net 分割模型（TDTS 训练）
```

| 权重文件 | 说明 | 论文对应 |
|---|---|---|
| `tdts_crm.pth` | Clean Recognition Model，在无噪声单字符集上训练 | 第 5.2.1 节；Table 3 |
| `tdts_nrm.pth` | Noisy Recognition Model，在加噪单字符集上训练 | 第 5.2.1 节；Table 3 |
| `tdts_mrm.pth` | Mixed Recognition Model，混合 CRM + NRM 训练集，论文主实验识别模型 | 第 5.2.1 节；Table 3、4 |
| `cvl_cnn_improved.pth` | CVL 单字符识别 CNN，结构与 CRM/NRM/MRM 一致，在 CVL 单字符子集上重训 | 第 5.4.1 节；Table 10 |
| `unet_tdts.pt` | TinyUNet 分割模型，在 TDTS 像素级掩码上训练 | 第 4.4 节；第 5.5 节；Table 12 |

---

## 五、复现流程

按以下顺序运行脚本，即可从零复现论文所有实验结果。以下命令默认从项目根目录开始执行。

```bash
# 步骤 1：准备数据
# 1.1 解压 CVL 数据集
unzip data/cvl_single_char.zip
unzip data/cvl_double_char.zip

# 1.2 生成 TDTS 数据集（如果还没有）
python src/tdts_build/B-new3240-noise2img.py

# 1.3 生成 U-Net 训练掩码
python src/tdts_unet/mask3240_gen.py

# 步骤 2：训练识别模型
python src/recognizer_training/train_tdts_recognizer.py
python src/recognizer_training/train_cvl_recognizer.py

# 步骤 3：训练 U-Net
python src/tdts_unet/unet_seg_train.py

# 步骤 4：跑分割实验
# 4.1 TDTS 上的图论方法
cd src/tdts_pipelines/cut_regionTest
python batch_main_recog2.py
cd ../../..

# 4.2 TDTS 上的 U-Net
cd src/tdts_unet
python batch_main_unet_recog.py
cd ../..

# 4.3 CVL 上的图论方法
cd src/cvl_pipelines
python batch_main_recog2.py
cd ../..

# 4.4 CVL 上的 U-Net 零样本
cd src/cvl_unet
python batch_main_unet_recog_cvl.py
cd ../..
```

---

## 六、引用

如果本仓库对您的研究有帮助，请引用：

```bibtex
@article{zhang2025graph,
  title   = {Research on graph theory segmentation method and dataset construction for connected handwritten digits},
  author  = {Zhang, Zhilai and Zhang, Yang and Wang, Yang},
  journal = {PLOS ONE},
  year    = {2025},
  note    = {Under review}
}
```

---
## 七、许可证

本项目采用 **MIT 许可证**，详见根目录下的 `LICENSE` 文件。

```text
MIT License

Copyright (c) 2025 Zhilai Zhang, Yang Zhang, Yang Wang
```

在符合 MIT 许可证条款的前提下，你可以自由使用、复制、修改、合并、发布、分发、再许可和/或销售本软件的副本。

---
## 八、联系方式

- 通讯作者：zhangzl@hdc.edu.cn
- 单位：邯郸学院信息工程学院
