#s2_place_coords_test.py
#本代码生成测试集的右侧图片的左上角坐标
#本程序生成在(14,0) ~(19,11)之间均分分布的点，一共10000个，保存在csv文件中
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = PROJECT_ROOT / 'data' / 'tdts_build_inputs' / 'coordinates_tst.csv'
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# 设置随机数种子，保证每次运行结果相同
np.random.seed(42)

# 定义 x 和 y 的取值范围
x_options = np.arange(14, 20)#  # 14 到 19（不包括 20）
y_options = np.arange(0, 12)#  # 0 到 11（不包括 12）

# 生成 10000 个坐标点，测试集10000个
num_points = 10000
x_values = np.random.choice(x_options, num_points)
y_values = np.random.choice(y_options, num_points)

# 创建 DataFrame
data = {
    'x': x_values,
    'y': y_values
}
df = pd.DataFrame(data)

# 保存到 CSV 文件
df.to_csv(OUT_CSV, index=False)

print(f"已成功生成 {num_points} 个坐标点，保存到 {OUT_CSV}")

# 统计每个坐标点的出现次数
df['coordinate'] = df.apply(lambda row: (row['x'], row['y']), axis=1)
coordinate_counts = df['coordinate'].value_counts(normalize=True)

# 输出每个坐标点的概率
print("各坐标点出现的概率：")
for coord, prob in coordinate_counts.items():
    print(f"坐标点 {coord}: {prob * 100:.2f}%")
