markdown
# Data Directory

This directory contains all input datasets, intermediate data, and benchmark
data used in the paper:

> **Research on Graph Theory Segmentation Method and Dataset Construction
> for Connected Handwritten Digits** (PLOS ONE, under revision)

The data are organized into three tiers: **build inputs**, **TDTS-derived
data**, and **CVL benchmark data**. A complete mapping between every data
directory and the corresponding table/section in the paper is given below.

---

## 1. Directory Overview
data/
├── mnist_compact_20x20/ Build input: 20×20 compact MNIST characters
├── tdts_build_inputs/ Build input: placement coordinates + pair indices
│
├── tdts_dataset/ TDTS touching two-digit image set (60k + 10k)
├── tdts_masks/ Pixel-level ground truth for U-Net training
├── tdts_single_char/ Single-character training sets for CRM/NRM/MRM
│ ├── clean/ → CRM training set
│ ├── noisy/ → NRM training set
│ └── mixed/ → MRM training set
│
├── cvl_single_char.zip CVL single-character set (packed, see §4)
├── cvl_double_char.zip CVL two-digit touching set (packed, see §4)
└── README.md This file

text

---

## 2. Data Description

### 2.1 Build Inputs

| Directory | Contents | Paper Reference |
|---|---|---|
| `mnist_compact_20x20/` | 20×20 compact-stroke MNIST characters extracted from the original MNIST images. Used as the fundamental building block of TDTS. | Section 3.3.1 |
| `tdts_build_inputs/` | Four CSV files controlling TDTS generation: right-character placement coordinates and left/right character-pair indices for both train and test splits. | Sections 3.2.1, 3.3.2, 3.3.3 |

### 2.2 TDTS-Derived Data

| Directory | Contents | Paper Reference |
|---|---|---|
| `tdts_dataset/` | The TDTS two-digit touching image set: **60,000** training-source images (TDTS-TR) and **10,000** test-source images (TDTS-TS), stored in MNIST idx-ubyte format (32×40 resolution). | Tables 1, 2, 4, 8, 9 |
| `tdts_masks/` | Pixel-level ground truth masks (`train-masks.npy`, `test-masks.npy`) for U-Net training. Each pixel is labeled as background (0), left character (1), or right character (2). | Section 4, Table 4 (U-Net row) |
| `tdts_single_char/` | Single-character training sets for three recognition models. Subdirectories `clean/`, `noisy/`, and `mixed/` correspond to **CRM**, **NRM**, and **MRM**, respectively. | Table 3 |
| `tdts_single_char/clean/` | Clean single-character training set (zero-padding normalized). Trains the Clean Recognition Model (CRM). | Table 3 |
| `tdts_single_char/noisy/` | Noisy single-character training set. Trains the Noisy Recognition Model (NRM). | Table 3 |
| `tdts_single_char/mixed/` | Union of `clean/` and `noisy/`. Trains the Mixed Recognition Model (MRM). MRM is used as the fixed recognition benchmark in all subsequent experiments. | Tables 3, 4 |

### 2.3 CVL Benchmark Data

The CVL data are distributed as compressed archives (see §4). After
extraction, each archive expands into a directory containing PNG files.

| Archive | Contents | Paper Reference |
|---|---|---|
| `cvl_single_char.zip` | CVL single-character set: 7,000 training images and 21,780 test images. Used to retrain the CVL recognition model (structurally identical to CRM/NRM/MRM). | Table 10 |
| `cvl_double_char.zip` | CVL two-digit touching set: 1,953 images covering 75 digit-pair classes, manually selected from genuine touching patterns in CVL digit strings. Used for zero-shot generalization evaluation. | Tables 10, 11 |

---

## 3. Data Generation Pipeline

The complete TDTS construction pipeline is:
MNIST ──┐
├─→ mnist_compact_20x20/ ──┐
│ │
MNIST metadata ───────┘ │
├─→ tdts_dataset/ ──→ tdts_masks/
│ ──→ tdts_single_char/
tdts_build_inputs/ ──────────────────────────────┘

text

Reproduction commands are provided in the top-level `README.md`.

---

## 4. Compressed Archives

For convenience and to reduce repository size, the CVL datasets are
distributed as ZIP archives with the following checksums:

| Archive | Size | SHA-256 |
|---|---|---|
| `cvl_single_char.zip` | 18,972,921 bytes (≈18.1 MB) | `B59E42E368684CBE712ED3A087958DC3A0FC3E6C36A9A9228E0789DD981DE723` |
| `cvl_double_char.zip` | 1,750,534 bytes (≈1.67 MB) | `12BAFE4F90718B42B8E0B02BB949BF251EFA4441562C6663F7E6089176B315BF` |

### Extraction

The archives are cross-platform ZIP files. To extract:

**Linux / macOS**

```bash
unzip cvl_single_char.zip
unzip cvl_double_char.zip
Windows (PowerShell)

powershell
Expand-Archive -Path "cvl_single_char.zip" -DestinationPath "."
Expand-Archive -Path "cvl_double_char.zip" -DestinationPath "."
After extraction, the directory structure should be:

text
data/
├── cvl_single_char/
│   ├── train/          ← 7,000 PNG images
│   └── test/           ← 21,780 PNG images
└── cvl_double_char/    ← 1,953 PNG images named {label}-xxx.png
Integrity Verification
To verify integrity after download:

Linux / macOS

bash
sha256sum -c <<EOF
B59E42E368684CBE712ED3A087958DC3A0FC3E6C36A9A9228E0789DD981DE723  cvl_single_char.zip
12BAFE4F90718B42B8E0B02BB949BF251EFA4441562C6663F7E6089176B315BF  cvl_double_char.zip
EOF
Windows (PowerShell)

powershell
Get-FileHash cvl_single_char.zip -Algorithm SHA256
Get-FileHash cvl_double_char.zip -Algorithm SHA256
The output must match the values listed in §4.

5. Notes on Data Format
Image format: All character images are stored as 8-bit grayscale PNG
(lossless), so that pixel-level segmentation results are fully
reproducible. Do not convert to JPEG or other lossy formats.

Resolution: TDTS and CVL images are both 32×40 pixels (height × width),
compatible with MNIST storage conventions.

Label encoding: Two-digit labels are encoded as L × 10 + R, where
L and R are the left and right character labels, respectively.
When L = 0, the label is encoded as 100 + R (see Section 3.3.4 of the
paper for details).

6. License
All data in this directory are released under CC BY 4.0. The underlying
MNIST and CVL datasets retain their original licenses and are used here only
as sources for generating derived datasets.

If you use these data in academic work, please cite the paper:

Zhang, Z., Zhang, Y., & Wang, Y. Research on Graph Theory Segmentation
Method and Dataset Construction for Connected Handwritten Digits.
PLOS ONE (under revision).

For any questions about the data, please contact the corresponding author
at zhangzl@hdc.edu.cn.