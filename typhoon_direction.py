import pandas as pd
import numpy as np
from tqdm import tqdm  # 用于进度条

# 读取数据
data = pd.read_csv('/media/data3/code/lzh/mathercup/dataset/september_typhoons2.csv')

# 遍历所有行
for index in tqdm(range(len(data) - 1), desc="计算移动方向"):  # 只遍历到倒数第二行
    
    
        lat_current = data.at[index, '纬度']
        lon_current = data.at[index, '经度']
        lat_next = data.at[index + 1, '纬度']
        lon_next = data.at[index + 1, '经度']

        # 计算变化
        delta_lat = lat_next - lat_current
        delta_lon = lon_next - lon_current

        # 计算方向角
        angle = np.degrees(np.arctan2(delta_lat, delta_lon))  # 使用 arctan2 计算角度
        angle = angle % 360  # 将角度标准化到 [0, 360)

        # 根据角度确定方向
        if (angle >= 337.5) or (angle < 22.5):
            direction = '南'
        elif 22.5 <= angle < 67.5:
            direction = '东北'
        elif 67.5 <= angle < 112.5:
            direction = '西'
        elif 112.5 <= angle < 157.5:
            direction = '西北'
        elif 157.5 <= angle < 202.5:
            direction = '北'
        elif 202.5 <= angle < 247.5:
            direction = '西南'
        elif 247.5 <= angle < 292.5:
            direction = '东'
        elif 292.5 <= angle < 337.5:
            direction = '东南'

        # 填充当前行的移动方向
        data.at[index, '移动方向'] = direction

# 保存更新后的 DataFrame
data.to_csv('/media/data3/code/lzh/mathercup/dataset/september_typhoons.csv', index=False)
