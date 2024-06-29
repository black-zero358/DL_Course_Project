import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# 从本地csv文件加载Iris数据集
from torch.utils.data import TensorDataset, DataLoader

data = pd.read_csv('iris/iris.data')
print(data)
# 数据预处理
X = data.iloc[:, :4].values  # 获取特征
Y = data.iloc[:, 4].values  # 获取标签

# 建立标签映射字典
# y_set = dict('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')
label_map = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
# label_map = {label: idx for idx, label in enumerate(y_set)}
# print(label_map)
#
# # 将y标签转换为0、1、2
# y_numeric = [float(label_map[label]) for label in Y]
# print("\ny转换为数值后的结果:")
# print(y_numeric)
#
# Y = torch.tensor(y_numeric)
Y_num=[]
for i in range(len(Y)):
    Y_num.append(label_map[Y[i]])

X=torch.tensor(X, dtype=torch.float32)
Y=torch.tensor(Y_num, dtype=torch.long)

print(len(X))
print(len(Y))

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rate = 0.7
train_len = int(rate * len(X))
trainX, trainY = X[:train_len], Y[:train_len]
# print(trainX)
# print(trainY)
testX, testY = X[train_len:], Y[train_len:]

batch_size = 50  # 设置包的大小（规模）
# 对训练集打包：
train_set = TensorDataset(trainX, trainY)
train_loader = DataLoader(dataset=train_set,  # 打包
                          batch_size=batch_size,  # 设置包的大小
                          shuffle=False)  # 默认：shuffle=False
# 对测试集打包：
test_set = TensorDataset(testX, testY)
test_loader = DataLoader(dataset=test_set,
                         batch_size=batch_size,  # 设置包的大小
                         shuffle=False)  # 默认shuffle=False


# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 3)  # 输出层3类

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.float()  # 将输出转换为Float类型

model = NeuralNetwork()


model = NeuralNetwork()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(10):  #训练代数设置为10
    for x, y in train_loader: #使用上面打包的训练集进行训练
        pre_y = model(x)
        loss=nn.CrossEntropyLoss()(pre_y,y)

        optimizer.zero_grad()  	#梯度清零
        loss.backward()       	#反向计算梯度
        optimizer.step()       	#参数更新
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
X_test_tensor = torch.tensor(testX, dtype=torch.float32)
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

accuracy = (predicted == torch.tensor(testY)).sum().item() / len(testY)
print(f'测试集准确率：{accuracy * 100:.2f}%')
