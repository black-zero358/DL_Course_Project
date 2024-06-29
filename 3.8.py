import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        # 第一层：输入层到第一个隐藏层
        self.fc1 = nn.Linear(3, 4)  # 输入维度为3，输出维度为4
        self.relu1 = nn.ReLU()  # 使用ReLU激活函数

        # 第二个隐藏层
        self.fc2 = nn.Linear(4, 3)  # 输入维度为4（上一层的输出），输出维度为3
        self.tanh2 = nn.Tanh()  # 使用tanh激活函数

        # 第三层：第二个隐藏层到输出层
        self.fc3 = nn.Linear(3, 2)  # 输入维度为3（上一层的输出），输出维度为2
        self.sigmoid3 = nn.Sigmoid()  # 使用sigmoid激活函数

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.tanh2(x)

        x = self.fc3(x)
        x = self.sigmoid3(x)

        return x


# 创建网络实例
net = CustomNet()

# 打印网络结构
print(net)