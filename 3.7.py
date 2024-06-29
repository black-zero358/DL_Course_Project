import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 加载数据集
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
data = pd.read_csv(data_url, header=None)

# 数据预处理
# 将分类变量转换为数值
label_encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = label_encoder.fit_transform(data[col])

# 划分特征和标签
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 转换为张量
train_X_tensor = torch.tensor(X_train, dtype=torch.float32)
train_Y_tensor = torch.tensor(y_train, dtype=torch.long)
test_X_tensor = torch.tensor(X_test, dtype=torch.float32)
test_Y_tensor = torch.tensor(y_test, dtype=torch.long)

# 将数据移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_X_tensor = train_X_tensor.to(device)
train_Y_tensor = train_Y_tensor.to(device)
test_X_tensor = test_X_tensor.to(device)
test_Y_tensor = test_Y_tensor.to(device)

# 构建数据集和数据加载器
train_set = TensorDataset(train_X_tensor, train_Y_tensor)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_set = TensorDataset(test_X_tensor, test_Y_tensor)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(14, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 输出层2类（binary classification）

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 实例化模型、定义损失函数和优化器，并将模型移动到GPU
model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 40
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2f}")
