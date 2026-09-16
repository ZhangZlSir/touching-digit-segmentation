#dataset_viewer.py
"""
Quick visualization of the TDTS dataset.
Displays a few sample images to help users verify the data.

Usage:
    python src/common/dataset_viewer.py
"""
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / 'data' / 'mnist_compact_20x20'

# Alternative roots for inspecting other datasets:
#DATA_clean = PROJECT_ROOT / 'data' / 'tdts_single_char' / 'clean'
#DATA_mixed = PROJECT_ROOT / 'data' / 'tdts_single_char' / 'mixed'
#DATA_noisy = PROJECT_ROOT / 'data' / 'tdts_single_char' / 'noisy'
#DATA_tdts_dataset = PROJECT_ROOT / 'data' / 'tdts_dataset'

img_data = MNIST(str(DATA_ROOT) , download=False,train=True)

for ii in [1,len(img_data)//4,len(img_data)//2,len(img_data)//2+len(img_data)//4,len(img_data)-1]:
    a_data, a_label = img_data[ii]
    plt.imshow(a_data, cmap='gray')
    plt.show() 
    print(a_label)
print(f"Total number of images in the dataset: {len(img_data)}")

