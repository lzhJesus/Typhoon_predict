import pandas as pd

# 读取数据
data = pd.read_csv('/media/data3/code/lzh/mathercup/dataset/september_typhoons.csv')

# 定义有效方向
valid_directions = {'北', '东北', '东', '东南', '南', '西南', '西', '西北'}

# 检查并修正移动方向
def check_direction(direction):
    if direction in valid_directions:
        return direction
    else:
        return direction[:2]  # 保留前两个字

# 应用检查函数
data['移动方向'] = data['移动方向'].apply(check_direction)

# 保存更新后的 DataFrame
data.to_csv('/media/data3/code/lzh/mathercup/dataset/september_typhoons2.csv', index=False)
