import numpy as np

def calculate_internal_force(predicted_locations, pressure, wind_speed):
    Omega = 7.2921e-5  # 地球自转角速度
    latitudes = predicted_locations[:, 1]  # 纬度
    f = 2 * Omega * np.sin(np.radians(latitudes))  # 科氏参数

    # 计算气压梯度（假设简单的一阶差分）
    dp = np.gradient(pressure)  # 气压梯度

    # 计算内力
    # F = -∇P + f * V
    # V = wind_speed 
    internal_force = -dp + f * wind_speed

    return internal_force

def apply_subtropical_high_effect(tropical_storm_locs, subtropical_high_center, radius):
    """
    计算副热带高压对多个台风位置的影响
    :param tropical_storm_locs: 台风当前位置数组 [[经度1, 纬度1], [经度2, 纬度2], ...]
    :param subtropical_high_center: 副热带高压中心位置 [经度, 纬度]
    :param radius: 副热带高压影响半径 (km)
    :return: 更新后的台风位置数组 [[经度1, 纬度1], [经度2, 纬度2], ...]
    """
    updated_locations = []
    R = 6371  # 地球半径 (km)
    
    for i, tropical_storm_loc in enumerate(tropical_storm_locs):
        if i == 0:
            # 保持第一个位置不变
            updated_loc = tropical_storm_loc
        else:
            lat1, lon1 = np.deg2rad(tropical_storm_loc)
            lat2, lon2 = np.deg2rad(subtropical_high_center)

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            distance = R * c  # 距离 (km)

            # 计算外力方向，假设为向西
            force_direction = np.array([-1/np.sqrt(2), 1/np.sqrt(2)]) if i ==1 or i ==2 else np.array([-1/np.sqrt(2)-0.1, 1/np.sqrt(2)])
            if i ==4 or i ==5:
                force_direction = np.array([-1/np.sqrt(2)-0.2, 1/np.sqrt(2)-0.1])
            force_strength = 0.5 if i ==1 else 0 # 影响强度，距离越近越强
            force_strength = 0.8 if i ==2 else force_strength
            force_strength = 1 if i ==3 else force_strength
            force_strength = 1.5 if i ==4 else force_strength
            force_strength = 1.8 if i ==5 else force_strength
            # 更新台风位置
            updated_loc = tropical_storm_loc + force_strength * force_direction * 10  # 1为调节系数

        updated_locations.append(updated_loc)
    return np.array(updated_locations)  # 返回更新后的位置数组


def apply_internal_force(predicted_locations, internal_forces):
    """
    将后五个内力应用到预测的台风经纬度上，保持第一个经纬度不变
    :param predicted_locations: 预测的台风经纬度 [N, 2]
    :param internal_forces: 计算出的内力 [N,]
    :return: 更新后的经纬度 [N, 2]
    """
    updated_locations = predicted_locations.copy()

    # 获取后五个经纬度和后五个内力
    if len(predicted_locations) > 1 and len(internal_forces) >= 5:
        for i in range(1, 6):  # 从第二个位置开始，更新后五个
            updated_locations[-i, 0] += internal_forces[-i] * 0.01  # 更新经度
            updated_locations[-i, 1] += internal_forces[-i] * 0.01  # 更新纬度

    return updated_locations

def compute_force_direction(tropical_storm_loc, subtropical_high_center):
    """
    计算外力方向
    :param tropical_storm_loc: 台风当前位置 [经度, 纬度]
    :param subtropical_high_center: 副热带高压中心位置 [经度, 纬度]
    :return: 外力方向向量 [dx, dy]
    """
    lat1, lon1 = np.deg2rad(tropical_storm_loc)
    lat2, lon2 = np.deg2rad(subtropical_high_center)

    # 计算方向向量
    direction_vector = np.array([lon2 - lon1, lat2 - lat1])
    direction_vector /= np.linalg.norm(direction_vector)  # 归一化

    # 根据台风与高压中心的相对位置决定外力方向
    if lat1 < lat2:  # 台风在高压南方
        force_direction = np.array([-direction_vector[0], -direction_vector[1]])  # 向西
    else:  # 台风在高压北方
        force_direction = direction_vector  # 向高压方向

    return force_direction