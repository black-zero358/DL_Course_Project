from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为(224,224)
    transforms.ToTensor(),  # 转化张量
])

# ====数据集====
class cat_dog_dataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.files = os.listdir(dir)

    def __len__(self):  # 需要重写该方法，返回数据集大小
        t = len(self.files)
        return t

    def __getitem__(self, idx):
        file = self.files[idx]
        fn = os.path.join(self.dir, file)
        img = Image.open(fn).convert('RGB')
        img = transform(img)  # 调整图像形状为(3,224,224), 并转为张量
        img = img.reshape(-1, 224, 224)
        y = 0 if 'cat' in file else 1  # 构造图像的类别
        return img, y


# =============================================
batch_size = 20
train_dir = './data/catdog/training_set2'  # 训练集所在的目录
test_dir = './data/catdog/test_set'  # 测试集所在的目录
train_dataset = cat_dog_dataset(train_dir)  # 创建数据集
train_loader = DataLoader(dataset=train_dataset,  # 打包
                          batch_size=batch_size,
                          shuffle=True)
test_dataset = cat_dog_dataset(test_dir)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)
print('训练集大小：', len(train_loader.dataset))
print('测试集大小：', len(test_loader.dataset))


# =====模型====
vgg16 = models.vgg16(pretrained=True).to(device)

conv1 = nn.Conv2d(3, 3, 3)  # (1, 3)，新定义
conv2 = vgg16.features[0]  # (3, 64)，来自 VGG16，参数需要冻结
conv3 = vgg16.features[2]  # (64, 64)，来自 VGG16，参数需要冻结
conv4 = nn.Conv2d(64, 512, 3)  # (64, 512)，新定义
conv5 = vgg16.features[28]  # (512, 512)，来自 VGG16，参数需要冻结

L = [conv2, conv3, conv5]  # 对这些网络层上的参数进行冻结
for layer in L:
    for param in layer.parameters():
        param.requires_grad = False


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.conv5 = conv5
        # 全连接层
        self.fc1 = nn.Linear(512 * 6 * 6, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):  # torch.Size([16, 1, 224, 224])
        o = x
        o = self.conv1(o)  # torch.Size([16, 3, 222, 222])
        o = nn.ReLU(inplace=True)(o)
        o = nn.MaxPool2d(2, 2)(o)  # torch.Size([16, 3, 111, 111])

        o = self.conv2(o)
        o = nn.ReLU(inplace=True)(o)
        o = nn.MaxPool2d(2, 2)(o)

        o = self.conv3(o)
        o = nn.ReLU(inplace=True)(o)
        o = nn.MaxPool2d(2, 2)(o)

        o = self.conv4(o)
        o = nn.ReLU(inplace=True)(o)
        o = nn.MaxPool2d(2, 2)(o)

        o = self.conv5(o)
        o = nn.ReLU(inplace=True)(o)
        o = nn.MaxPool2d(2, 2)(o)

        o = o.reshape(x.size(0), -1)

        o = self.fc1(o)  # 全连接层
        o = nn.ReLU(inplace=True)(o)
        o = nn.Dropout(p=0.5, inplace=False)(o)
        o = self.fc2(o)  # 全连接层
        o = nn.ReLU(inplace=True)(o)
        o = nn.Dropout(p=0.5, inplace=False)(o)
        o = self.fc3(o)  # 全连接层
        return o

# ====测试模型结构====

net = Net().to(device)
x = torch.randn(16, 3, 224, 224).to(device)  # 随机产生测试数据
y = net(x)  # 调用网络模型

param_sum = 0  # 统计参数总数
trainable_param_sum = 0  # 统计可训练的参数总数
for param in net.parameters():
    n = 1
    for j in range(len(param.shape)):  # 统计当前层的参数个数
        n = n * param.size(j)
    param_sum += n
    if param.requires_grad:
        trainable_param_sum += n
print('该模型的参数总数为：{:.0f}，其中可训练的参数总数为：\
      {:.0f}，占的百分比为：{:.2f}%'. \
      format(param_sum, trainable_param_sum, \
             100. * trainable_param_sum / param_sum))

print('输入和输出的形状分别为：', x.shape, y.shape)


# ====训练
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

start = time.time()  # 开始计时
net.train()
for epoch in range(10):  # 执行10代
    ep_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pre_y = net(x)
        loss = nn.CrossEntropyLoss()(pre_y, y.long())  # 使用交叉熵损失函数
        ep_loss += loss * x.size(0)  # loss是损失函数的平均值,故要乘以样本数量
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('第 %d 轮循环中，损失函数的平均值为: %.4f' \
          % (epoch + 1, (ep_loss / len(train_loader.dataset))))
end = time.time()  # 计时结束
print('训练时间为:  %.1f 秒 ' % (end - start))

# ====测试

correct = 0
net.eval()
with torch.no_grad():
    for i, (x, y) in enumerate(train_loader):  # 计算在训练集上的准确率
        x, y = x.to(device), y.to(device)
        pre_y = net(x)
        pre_y = torch.argmax(pre_y, dim=1)
        t = (pre_y == y).long().sum()
        correct += t
t = 1. * correct / len(train_loader.dataset)
print('1、网络模型在训练集上的准确率：{:.2f}%' \
      .format(100 * t.item()))

correct = 0
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):  # 计算在测试集上的准确率
        x, y = x.to(device), y.to(device)
        pre_y = net(x)
        pre_y = torch.argmax(pre_y, dim=1)
        t = (pre_y == y).long().sum()
        correct += t
t = 1. * correct / len(test_loader.dataset)
print('2、网络模型在测试集上的准确率：{:.2f}%' \
      .format(100 * t.item()))