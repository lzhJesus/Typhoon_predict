import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model_predict import LSTMModel
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataset_encoder import dataset_encoder

def train(model, train_loader, num_epochs, learning_rate):
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    losses = []  # 记录损失的列表
    for epoch in range(num_epochs):
        epoch_loss = 0  # 每个 epoch 的总损失
        for src, tgt in train_loader:
            optimizer.zero_grad()
            output = model(src.unsqueeze(1))  # 添加序列维度
            loss = criterion(output, tgt)  # tgt 是下一个时间步的标签
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)  # 计算平均损失
        losses.append(avg_loss)  # 记录损失
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')

    # 绘制训练损失图
    plt.plot(losses)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('/media/data3/code/lzh/Bebincia_predict/dataset/training_loss.png')  # 保存图像


def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for src, tgt in test_loader:
            output = model(src.unsqueeze(1))  # 添加序列维度
            loss = criterion(output, tgt)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss}')

    
if __name__ == "__main__":

    scaler_reverse = StandardScaler()
    scaler_normal = StandardScaler()
    onehotencoder = OneHotEncoder(sparse=False)
    print("dataset_encoder processing")
    dataset_encoder('/media/data3/code/lzh/Bebincia_predict/dataset/september_typhoons.csv', scaler_reverse, scaler_normal, onehotencoder)
    print("data reading")
    # 加载数据
    data = pd.read_csv('/media/data3/code/lzh/Bebincia_predict/dataset/typhoons_byencoder.csv')

    # 假设特征列为 features，目标列为经纬度
    features = data[['经度', '纬度', '风速', '台风等级', '气压', '移动速度', 
                    '天', '小时', '台风强度_台风(TY)', '台风强度_强台风(STY)', 
                    '台风强度_强热带风暴(STS)', '台风强度_热带低压(TD)', 
                    '台风强度_热带风暴(TS)', '台风强度_超强台风(Super TY)', 
                    '移动方向_东', '移动方向_东北', '移动方向_东南', 
                    '移动方向_北', '移动方向_南', '移动方向_西', 
                    '移动方向_西北', '移动方向_西南']].values

    # 创建目标列，假设是下一个时间步的经纬度
    target = data[['经度', '纬度']].shift(-1).fillna(0).values  # 使用下一个时间步的经纬度作为目标

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # 将数据转换为 Tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    input_dim = features.shape[1]
    hidden_dim = 128
    output_dim = 2
    num_layers = 3

    # 创建模型
    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
    # 开始训练
    train(model, train_loader, num_epochs=500, learning_rate=0.001)
    # 评估模型
    evaluate(model, test_loader)
    # 模型保存
    torch.save(model.state_dict(), '/media/data3/code/lzh/Bebincia_predict/modelpth/model.pth')
    joblib.dump(scaler_reverse, '/media/data3/code/lzh/Bebincia_predict/modelpth/scaler_reverse.pkl')
    joblib.dump(scaler_normal, '/media/data3/code/lzh/Bebincia_predict/modelpth/scaler_normal.pkl')
    joblib.dump(onehotencoder, '/media/data3/code/lzh/Bebincia_predict/modelpth/onehot_encoder.pkl')
    print("Model and scaler saved.")
