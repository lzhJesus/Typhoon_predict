import numpy as np
import torch
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model_predict import LSTMModel
import matplotlib.pyplot as plt
from utils_typhoon import calculate_internal_force, apply_internal_force, apply_subtropical_high_effect

def dtw(sequence_a, sequence_b):
    """
    Compute the Dynamic Time Warping (DTW) distance between two sequences.
    
    Parameters:
    sequence_a (np.ndarray): First sequence of shape (n, m)
    sequence_b (np.ndarray): Second sequence of shape (k, m)
    
    Returns:
    float: The DTW distance between the two sequences.
    """
    n = len(sequence_a)
    k = len(sequence_b)
    
    # Create the cost matrix
    dtw_matrix = np.zeros((n + 1, k + 1))
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf
    
    # Compute DTW matrix
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            cost = np.linalg.norm(sequence_a[i - 1] - sequence_b[j - 1])  # Calculate the cost (Euclidean distance)
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # Insertion
                                           dtw_matrix[i, j - 1],    # Deletion
                                           dtw_matrix[i - 1, j - 1])  # Match
    
    return dtw_matrix[n, k]

# 加载模型
model = LSTMModel(22, 128, 2, 3)  # 需先定义模型结构
model.load_state_dict(torch.load('/media/data3/code/lzh/Bebincia_predict/modelpth/model.pth'))
model.eval()  # 设置为评估模式

# 加载标准化器
scaler_reverse = joblib.load('/media/data3/code/lzh/Bebincia_predict/modelpth/scaler_reverse.pkl')
scaler_normal = joblib.load('/media/data3/code/lzh/Bebincia_predict/modelpth/scaler_normal.pkl')

# 加载独热编码器
onehot_encoder = joblib.load('/media/data3/code/lzh/Bebincia_predict/modelpth/onehot_encoder.pkl')

# 输入数据字典
data_dict = {
    '台风强度': ['强热带风暴(STS)', '强热带风暴(STS)', '强热带风暴(STS)', '台风(TY)', '台风(TY)', '热带风暴(TS)'],
    '移动方向': ['西北', '西北', '西北', '西北', '西北', '西北'],
    '纬度': [19.1, 11.0, 11.0, 13.0, 12.0, 8.0],
    '经度': [139.8, 30.0, 30.0, 40.0, 33.0, 18.0],
    '台风等级': [11.0, 10.0, 11.0, 13.0, 12.0, 8.0],
    '风速': [30.0, 28.0, 30.0, 40.0, 33.0, 18.0],
    '气压': [980.0, 982.0, 980.0, 960.0, 975.0, 998.0],
    '移动速度': [25, 32, 30, 25, 15, 14],
    '天': [12, 13, 14, 15, 16, 17],
    '小时': [14, 14, 14, 14, 14, 14]
}

# 初始化 DataFrame
input_data = pd.DataFrame(data_dict)

# 循环预测
predicted_locations = input_data[['经度', '纬度']].values.copy()

for i in range(len(input_data)):
    # 当前输入数据
    current_data = input_data.iloc[i:i + 1]

    # 对分类属性进行独热编码
    encoded_features = onehot_encoder.transform(current_data[['台风强度', '移动方向']])

    # 对经纬度进行标准化
    lat_lon_features = current_data[['经度', '纬度']]
    scaled_lat_lon_features = scaler_reverse.transform(lat_lon_features)

    # 对其他数值特征进行标准化
    numerical_features = ['风速', '台风等级', '气压', '移动速度', '天', '小时']
    scaled_other_numerical_features = scaler_normal.transform(current_data[numerical_features])

    # 合并经纬度的标准化特征和其他数值特征
    final_input = np.hstack((scaled_lat_lon_features, scaled_other_numerical_features, encoded_features))

    # 将数据转换为 Tensor
    input_tensor = torch.FloatTensor(final_input)

    # 进行预测
    with torch.no_grad():
        prediction = model(input_tensor.unsqueeze(1))  # 添加序列维度
    
    # 反标准化经纬度
    decoded_prediction = scaler_reverse.inverse_transform(prediction.numpy())

    
    # 将预测的经纬度添加到下一个输入
    if i + 1 < len(input_data):
        input_data.at[input_data.index[i + 1], '经度'] = decoded_prediction[0][0]
        input_data.at[input_data.index[i + 1], '纬度'] = decoded_prediction[0][1]
        predicted_locations[i+1] = decoded_prediction[0]

# 转换为 NumPy 数组
predicted_locations = np.array(predicted_locations).squeeze()
pressure = np.array([980.0, 982.0, 980.0, 960.0, 975.0, 998.0])  # 气压数据
wind_speed = np.array([30.0, 28.0, 30.0, 40.0, 33.0, 18.0])  # 风速
typhoon_internal_force = calculate_internal_force(predicted_locations, pressure, wind_speed)
predicted_locations = apply_internal_force(predicted_locations, typhoon_internal_force)
subtropical_high_center = [140.0, 25.0]  # 经度, 纬度
radius = 800  # 影响半径 (km)
predicted_locations = apply_subtropical_high_effect(predicted_locations, subtropical_high_center, radius)
actual_locations = np.array([
    [139.8,19.1],
    [135.6,23.6],
    [130.3,27.3],
    [126.2,30.10],
    [120.5,31.5],
    [116.3,33.0]
])
# 计算 DTW 距离
distance = dtw(actual_locations, predicted_locations)

# # 打印 DTW 距离
print(f"DTW 距离: {distance}")

# # 可视化结果
plt.plot(actual_locations[:, 0], actual_locations[:, 1], marker='o', label='Actual', color='blue')
plt.plot(predicted_locations[:, 0], predicted_locations[:, 1], marker='x', label='Predict', color='red')
plt.title('Bebincia route comparison')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()  # 添加图例
plt.savefig('/media/data3/code/lzh/Bebincia_predict/dataset/route_comparison.png')  # 保存图像
# 将结果保存成excel文件
results_df = pd.DataFrame({
    '实际经度': actual_locations[:, 0],
    '实际纬度': actual_locations[:, 1],
    '预测经度': predicted_locations[:, 0],
    '预测纬度': predicted_locations[:, 1],
    'DTW距离': distance
})

# 保存到 Excel 文件
results_df.to_excel('/media/data3/code/lzh/Bebincia_predict/dataset/typhoon_predictions.xlsx', index=False)

print("结果已保存到 typhoon_predictions.xlsx")





