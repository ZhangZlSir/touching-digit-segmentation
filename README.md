# 连通手写数字的图论分割方法与数据集构建

本仓库是论文 **“Research on graph theory segmentation method and dataset construction for connected handwritten digits”**（PLOS ONE 21(10): e0360920）的配套代码，包含方法实现、数据构建流程与实验脚本。

> **数据与模型权重不随仓库发布。** 数据集由本仓库脚本从 MNIST 生成，权重由训练脚本生成；
> 随仓库提供的只有 2 个 CVL 数据压缩包（见 §2），其余数据与权重均可由脚本重新生成。

## 论文信息

| 项目 | 内容 |
|---|---|
| 标题 | Research on graph theory segmentation method and dataset construction for connected handwritten digits |
| 作者 | Zhilai Zhang, Yang Zhang, Yang Wang |
| 期刊 | PLOS ONE **21(10): e0360920**（2026） |
| DOI | https://doi.org/10.1371/journal.pone.0360920 |
| 通讯作者 | zhangzl@hdc.edu.cn |

## 核心贡献

**1. 一种近似无阈值的切割域（cutting domain）生成方法。**
在实际图像上，切割域的**中心点**由「字符内容的几何中心」与「笔画投影的质量中心（质心）」取平均确定：

```text
L, R        = 最长连续非零投影区间的左右端点（自动排除孤立噪点列）
geo_center  = 0.5 × (L + R)
mass_center = Σ(x · proj[x]) / Σ proj[x],  x ∈ [L, R]
center      = 0.5 × (geo_center + mass_center)
r           = margin_frac × (R − L)
corridor    = clip([center − r, center + r], L, R)
```

对使用者而言，整个流程**只需设定一个参数** `margin_frac`——切割域半径占内容宽度的**相对比例**，
而不是像素阈值，因此对不同字符尺度自动适配，无需按图逐张调参（其余为固定结构常量，见源码注释）；
域内的生长同样采用「低灰度/背景优先」的无阈值规则。

这一设计解决的是图论分割在真实场景中的落地障碍：切割域一旦需要阈值或经验参数，Dijkstra、A\* 等
最优路径搜索就无法直接使用。方法的主要效果是**让图论分割方法可以直接用于实际图像，避免复杂的参数调整**；
该方法的有效性在自建的 **TDTS** 与 **CVL** 两套双字符集上得到验证。

实现位置：`src/cvl_pipelines/corridor_utils.py`（`compute_corridor`，中心点与 `margin_frac`）、
`src/cvl_pipelines/cutting_domain.py`（域内无阈值生长）、`src/cvl_pipelines/path_search.py`（路径搜索）。

**2. TDTS 双字符粘连数据集及构建流程。**
覆盖 `00`–`99` 全部 100 类，以不超过 30% 的最大重叠比例与随机空间错位模拟真实粘连，
配合类别均衡采样，构成可控、可复现的评测基准（第 3 章；构建脚本见 §5.2）。
另从 CVL 真实手写数字串中整理出 1,953 张双字符粘连样本，用于跨域评估。

**3. 图论路径搜索实现与系统性对比。**
在切割域上实现 Dijkstra 与 A\* 家族（Left-Biased、Right-Biased、Unbiased，含条件穿透机制），
并与垂直投影法（VPM）、约束分水岭（WA）、监督式 U-Net 在同一评测口径下比较（第 4–5 章）。

**验证结果**：TDTS-TS 上 Left-Biased A\* 端到端识别准确率 **93.13%**；CVL 上 Unbiased A\* 保持 **91.4%**；
监督式 U-Net 跨域从 **95.90%** 降至 **53.05%**，而无监督图论方法仅下降 **1.28** 个百分点。

---

## 1. 目录结构

```text
touching-digit-segmentation/
├── data/          CVL 数据压缩包（其余数据由脚本生成）
├── models/        模型权重目录（权重自行训练后放入）
├── results/       运行输出（历史逐批次结果 CSV 归档；统计表与插图可再生成，不纳入版本管理）
├── src/           全部源代码
├── requirements.txt
├── License
└── README.md
```

```text
src/
├── tdts_build/                 [第 3 章] TDTS 构建（按 s1 → s4 顺序执行）
│   ├── s1_mnist_compact_20x20.py    MNIST 28×28 → 去外框 → 20×20
│   ├── s2_place_coords_train.py     右字符坐标（训练集，60000 条）
│   ├── s2_place_coords_test.py      右字符坐标（测试集，10000 条）
│   ├── s3_balanced_pairs_train.py   左右配对表（训练集，每类均衡到 6000）
│   ├── s3_balanced_pairs_test.py    左右配对表（测试集，每类均衡到 1000；默认 seed=42）
│   ├── s3_seed_search_test.py       搜索使测试集标签对 MAD 最小的随机种子
│   └── s4_build_tdts.py             40×32 粘连图生成（噪点 + 最大值融合）
├── single_char_build/          [5.2.1] 三个单字符识别库
│   ├── build_clean.py               CRM 训练/测试库（补零填充）
│   ├── build_noisy.py               NRM 训练/测试库（加噪）
│   └── build_mixed.py               MRM 混合库（noisy + clean 拼接）
├── dataset_stats/              [第 3 章统计，Table 1/2]
│   ├── char_size_stats.py           单字符笔画尺寸表
│   ├── adhesion_stats.py            矩形/实际粘连率分布（Table 2）
│   ├── balance_mad_cv.py            MAD/CV 与蒙特卡洛模拟（Table 1）
│   ├── label_distribution.py        标签分布统计
│   ├── theoretical_mad.py           双字符库理论 MAD
│   ├── double_char_size.py          双字符笔画尺寸
│   ├── pixel_sum_bins.py            像素累加分档（噪声强度依据）
│   └── verify_pairs_match_images.py 数据一致性检查（数据集 ↔ 配对表/坐标）
├── recognizer_training/
│   ├── train_tdts_cnn.py            CRM/NRM/MRM（--set clean|noisy|mixed）
│   └── train_cvl_cnn.py             CVL 单字符识别 CNN
├── tdts_pipelines/             [4.2–4.4, 5.2–5.3] TDTS 上的分割实验
│   ├── cut_regionTest/              本文方法（切割域 + 路径搜索）
│   ├── cut_line_all/                切割线与识别结果同时输出
│   ├── water_all/                   WA（约束分水岭）
│   └── deNoise_water/               分水岭 + 去噪
├── cvl_pipelines/              [5.4] CVL 上的图论分割
├── figures/                    [第 4 章插图] Fig 4/5/7/8/9 的绘图脚本
├── tdts_unet/                  [4.4, 5.5] TDTS 上的 U-Net
│   ├── build_masks.py               逐像素真值掩码
│   ├── train_unet.py                训练 TinyUNet
│   ├── unet_seg_model.py            网络定义
│   ├── evaluate.py                  U-Net 分割 + 识别
│   └── pipeline.py / cnn_base.py / denoise.py
├── cvl_unet/                   [5.5] CVL 上的 U-Net 零样本对比
└── common/                     通用工具
    ├── view_dataset.py              数据集可视检查
    └── merge_results.py             批次结果 CSV 合并
```

每个流水线子目录自带 `pipeline.py`、`path_search.py`、`denoise.py` 等依赖，因此
**运行前需 `cd` 进入该目录**（同级 `import` 以脚本所在目录为搜索路径）。

---

## 2. 数据

### 2.1 随仓库提供

仓库中只提供两套 CVL 数据压缩包（MNIST 为公开数据集，TDTS 及其中间数据由脚本生成）：

| 路径 | 内容 | 论文对应 |
|---|---|---|
| `data/cvl_single_char.zip` | CVL 单字符集：训练 7,000 / 测试 21,780 张 PNG | 5.4.1；Table 10 |
| `data/cvl_double_char.zip` | CVL 双字符粘连集：1,953 张，覆盖 75 类 | 5.4.2；Table 11 |

压缩包校验值：

| 压缩包 | 大小 | SHA-256 |
|---|---|---|
| `cvl_single_char.zip` | 18,972,921 B | `B59E42E368684CBE712ED3A087958DC3A0FC3E6C36A9A9228E0789DD981DE723` |
| `cvl_double_char.zip` | 1,750,534 B | `12BAFE4F90718B42B8E0B02BB949BF251EFA4441562C6663F7E6089176B315BF` |

### 2.2 由脚本生成

| 产物 | 生成脚本 |
|---|---|
| `data/mnist_compact_20x20/` | `src/tdts_build/s1_mnist_compact_20x20.py` |
| `data/tdts_build_inputs/` 四张 CSV | `src/tdts_build/s2_place_coords_*.py`、`s3_balanced_pairs_*.py` |
| `data/tdts_dataset/` | `src/tdts_build/s4_build_tdts.py` |
| `data/tdts_single_char/{clean,noisy,mixed}/` | `src/single_char_build/build_*.py` |
| `data/tdts_masks/` | `src/tdts_unet/build_masks.py` |

`data/tdts_build_inputs/` 下四张 CSV 的格式：`coordinates_*.csv` 为 `x, y` 两列；
`mnist_image_pairs_*.csv` 列名为 `Left Image Index, Right Image Index, Left Label, Right Label, Two - Digit Label`，
标签按 `L×10+R` 编码、当 `L=0` 时记为 `100+R`（论文 3.3.4），该转换由 `s4_build_tdts.py` 在生成图像时完成。
两张坐标文件与两张配对表分别固定 `seed=42`，可逐字节复现。

数据目录统一为 `<root>/MNIST/raw/{train,t10k}-{images-idx3,labels-idx1}-ubyte`，
`torchvision.datasets.MNIST(root)` 可直接读取（`raw` 为小写，Linux 下大小写敏感）。
`.gitignore` 已排除上述生成目录，clone 后不存在属正常。

---

## 3. 模型权重

权重不随仓库发布。评估脚本按固定文件名从 `models/` 读取，请先训练后放入：

| 文件名 | 说明 | 产出脚本 | 论文对应 |
|---|---|---|---|
| `tdts_crm.pth` | Clean Recognition Model（无噪单字符库） | `train_tdts_cnn.py --set clean` | Table 3 |
| `tdts_nrm.pth` | Noisy Recognition Model（加噪单字符库） | `train_tdts_cnn.py --set noisy` | Table 3 |
| `tdts_mrm.pth` | Mixed Recognition Model（混合库，主实验使用） | `train_tdts_cnn.py --set mixed` | Table 3、4 |
| `cvl_cnn_improved.pth` | CVL 单字符识别 CNN（结构同 CRM/NRM/MRM） | `train_cvl_cnn.py` | Table 10 |
| `unet_tdts.pt` | TinyUNet 分割模型（TDTS 训练，CVL 零样本复用） | `tdts_unet/train_unet.py` | Table 12 |

训练还会写出各轮检查点与 `state_dict` 版本，以及逐轮指标 CSV 与曲线图（均位于 `models/`）。

单字符识别模型的参考指标（论文 Table 3）：

| 模型 | TDTS-TR | TDTS-TS |
|---|---|---|
| CRM | 98.83% | 98.16% |
| NRM | 98.01% | 97.95% |
| MRM | 99.36% | 98.03% |

---

## 4. 安装

```bash
pip install -r requirements.txt
```

论文 5.1 的实验环境为 Python 3.9 + PyTorch 1.12.0（CPU，Intel i7-3630QM / 16 GB）；
代码亦在 Python 3.13 + PyTorch 2.9.1 下验证通过。

---

## 5. 复现流程

命令均从项目根目录开始；标注 `cd` 的步骤在对应子目录内执行。

### 5.1 解压 CVL 数据

```bash
cd data
mkdir -p cvl_single_char
unzip cvl_single_char.zip -d cvl_single_char   # 压缩包顶层即 trainStd40-32/ 与 evalStd40-32/
unzip cvl_double_char.zip                      # 压缩包顶层即 cvl_double_char/
cd ..
```

PowerShell：

```powershell
New-Item -ItemType Directory -Force data/cvl_single_char
Expand-Archive -Path data/cvl_single_char.zip -DestinationPath data/cvl_single_char
Expand-Archive -Path data/cvl_double_char.zip -DestinationPath data
```

### 5.2 构建 TDTS 数据集（第 3 章）

```bash
python src/tdts_build/s1_mnist_compact_20x20.py   # 首次运行由 torchvision 下载 MNIST 原始文件
python src/tdts_build/s2_place_coords_train.py
python src/tdts_build/s2_place_coords_test.py
python src/tdts_build/s3_balanced_pairs_train.py
python src/tdts_build/s3_balanced_pairs_test.py
python src/tdts_build/s4_build_tdts.py            # 40×32、最大重叠 30%、随机错位、噪点 20–120
```

### 5.3 构建三个单字符识别库（5.2.1）

```bash
python src/single_char_build/build_clean.py
python src/single_char_build/build_noisy.py
python src/single_char_build/build_mixed.py
```

### 5.4 生成 U-Net 训练掩码

```bash
python src/tdts_unet/build_masks.py
```

### 5.5 训练模型

```bash
python src/recognizer_training/train_tdts_cnn.py --set clean
python src/recognizer_training/train_tdts_cnn.py --set noisy
python src/recognizer_training/train_tdts_cnn.py --set mixed
python src/recognizer_training/train_cvl_cnn.py
python src/tdts_unet/train_unet.py
```

### 5.6 TDTS 上的分割实验（Tables 4–7）

```bash
cd src/tdts_pipelines/cut_regionTest
python evaluate.py --model mrm --split ts    # TDTS-TS
python evaluate.py --model mrm --split tr    # TDTS-TR
cd ../../..
```

`--model {crm,nrm,mrm}` 选择识别模型，`--split {tr,ts}` 选择数据源；
省略 `--start` / `--batch-size` 时按提示交互输入。`cut_line_all`、`water_all`、`deNoise_water`
三个目录用法相同，仅分割方法不同。

### 5.7 数据集统计（Table 1、Table 2）

```bash
python src/dataset_stats/char_size_stats.py   # 尺寸表，为 adhesion_stats 的前置
python src/dataset_stats/adhesion_stats.py    # Table 2
python src/dataset_stats/balance_mad_cv.py    # Table 1
```

### 5.8 U-Net 与 CVL 评估（Tables 10–12）

```bash
cd src/tdts_unet      && python evaluate.py --model mrm && cd ../..
cd src/cvl_pipelines  && python evaluate.py --yes       && cd ../..
cd src/cvl_unet       && python evaluate_zeroshot.py    && cd ../..
```

### 5.9 论文插图（Fig 4/5/7/8/9）

```bash
cd src/figures
python fig_region_growth.py   # Fig 7 切割域生长（14 步）
python fig_cut_lines.py       # Fig 8 六种算法的切割线对比
python fig_watershed.py       # Fig 9 约束分水岭分割结果
python fig_denoise.py         # Fig 4/5 原始粘连图与去噪结果
cd ../..
```

输出写入 `results/figures/`（300 dpi TIFF + PDF）。运行前建议：

```bash
export MPLBACKEND=Agg          # 无图形界面环境
export PYTHONIOENCODING=utf-8  # Windows 中文控制台，否则部分符号打印会报编码错误
```

- `src/figures/` 自带 5 个依赖模块（`denoising1.py`、`region_up.py`、`cut_region.py`、
  `path_generationSafe10.py`、`water12.py`），版本与 `src/tdts_pipelines/` 下的同名模块相互独立，
  目的是保证输出与论文插图一致。
- Fig 4/5 默认输出与论文插图内容一致（同样本、同去噪、同布局），仅画布尺寸略有差异；
  若需画布尺寸也一致，可用 `python fig_denoise.py --plos-exact`（输出 2220×900，论文为 2220×899）。
- Fig 1/2/3/6/10 为流程图/框架图（Visio 绘制），不含绘图代码；Fig 11 随 CVL 实验一并产生。

### 5.10 效率与分辨率扩展（Tables 8、9）

```bash
cd src/tdts_pipelines/cut_line_all
python bench_runtime.py          # Table 8：逐批次计时
python resolution_scaling.py     # Table 9：2×/4×/8× 分辨率
cd ../../..
```

---

## 6. 复现性说明

1. **随机种子**：`s2_place_coords_*.py` 与 `s3_balanced_pairs_*.py` 固定 `seed=42`，可逐字节复现
   `data/tdts_build_inputs/` 中的四张 CSV；`s3_balanced_pairs_*.py` 运行时会直接打印标签对的
   MAD 与 CV（TR：28.64 / 0.0631，TS：8.64 / 0.1110，即 Table 1 的均衡策略行）。
2. **单字符库（CRM / NRM / MRM）**：生成时未固定随机数种子——字符在 32×40 画布上的位置
   （`x∈[0,20]`、`y∈[0,12]`）与噪点位置、取值均为随机采样，因此重新运行 `build_*.py` 得到的是
   **同分布的新实例**，而非逐像素复现。**如需论文中使用的原始单字符库，请联系作者**
   （zhangzl@hdc.edu.cn）。当前三个脚本已固定 `random.seed(42)`，重跑结果稳定，可用于新的实验。
3. **切割域默认变体**：`src/tdts_pipelines/cut_regionTest/cutting_domain.py` 中
   `find_connected_points()` 为切割域构建入口，默认返回 **FD1**（`find_connected_points1`，
   8 邻域全连通、自上而下生长），与论文 4.2.2 一致。做 Table 6/7 消融时修改该行即可：
   `find_connected_points1Min`（FD2，重力偏置）、`find_row_optimal_points`（FD3，逐行最小值、无连通性约束）、
   `find_4neighbor_points`（FD4，4 邻域）、`find_bidirectional_points`（FD5，双向生长）。
4. **去噪消融**：`src/tdts_pipelines/cut_regionTest/pipeline.py` 中
   `denoised_image = remove_salt_pepper_noise(original_image)` 即去噪开关；注释该行、改用下一行的
   `denoised_image = original_image` 即为「不去噪」（Table 5）。
5. **分辨率扩展的样本数**：`resolution_scaling.py` 中 `N_SAMPLES = 50`；Table 9 使用 1000 张/分辨率，
   正式实验请调大。
6. **Table 8 的测量范围**：`bench_runtime.py` 记录逐批次运行时间；内存占用与路径环数需结合系统监控
   工具与输出的结果 CSV 统计。

---

## 7. 引用

```bibtex
@article{zhang2026graph,
  title   = {Research on graph theory segmentation method and dataset construction for connected handwritten digits},
  author  = {Zhang, Zhilai and Zhang, Yang and Wang, Yang},
  journal = {PLoS One},
  volume  = {21},
  number  = {10},
  pages   = {e0360920},
  year    = {2026},
  doi     = {10.1371/journal.pone.0360920}
}
```

---

## 8. 许可证

本项目采用 **MIT 许可证**，详见根目录 [`License`](License)。

```text
MIT License

Copyright (c) 2026 Zhilai Zhang, Yang Zhang, Yang Wang
```

在符合 MIT 许可证条款的前提下，可自由使用、复制、修改、合并、发布、分发、再许可和/或销售本软件的副本。

---

## 9. 联系方式

- 通讯作者：zhangzl@hdc.edu.cn
- 单位：邯郸学院信息工程学院
- 本仓库代码与数据仅供学术研究使用。
