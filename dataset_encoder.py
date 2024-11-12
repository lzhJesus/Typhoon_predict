import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def dataset_encoder(path, scaler_reverse, scaler_normal,encoder):
    data = pd.read_csv(path)

    # 对分类属性进行独热编码
    categorical_features = ['台风强度', '移动方向']
    encoded_features = encoder.fit_transform(data[categorical_features])

    # 将编码后的特征转换为 DataFrame
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

    # 选择数值特征并进行标准化
    numerical_features = ['风速', '台风等级', '气压', '移动速度', '天', '小时']
    scaled_numerical_features = scaler_normal.fit_transform(data[numerical_features])
    scaled_df = pd.DataFrame(scaled_numerical_features, columns=numerical_features)

    # 对经纬度进行反向标准化
    lat_lon_features = data[['经度', '纬度']]
    scaled_lat_lon_features = scaler_reverse.fit_transform(lat_lon_features)
    scaled_lat_lon_df = pd.DataFrame(scaled_lat_lon_features, columns=['经度', '纬度'])

    # 将编码和标准化后的特征合并
    final_data = pd.concat([scaled_lat_lon_df, scaled_df, encoded_df], axis=1)

    # 保存最终处理的数据
    final_data.to_csv('/media/data3/code/lzh/Bebincia_predict/dataset/typhoons_byencoder.csv', index=False)

