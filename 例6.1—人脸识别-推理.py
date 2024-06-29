import matplotlib.pyplot as plt
import torch
from torch import nn, threshold

import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设备选择，如果有可用的CUDA设备则使用CUDA，否则使用CPU

torch.manual_seed(123)
random.seed(123)
np.random.seed(10)
# 设置随机种子，使得实验结果可复现


# 定义图像预处理操作，包括转换为PIL图像、缩放、随机水平翻转、转换为张量、归一化
transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.Resize((100, 100)), transforms.RandomHorizontalFlip(p=0.5),
     transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


# 定义孪生网络模型
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.ReflectionPad2d(1),
            # ......
            nn.Conv2d(1, 3, 3, padding=1),
            vgg19_cnn, nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512)
        )

    def forward_once(self, x):
        o = x
        o = self.cnn(o)
        o = o.reshape(x.size(0), -1)
        o = self.fc1(o)
        return o

    def forward(self, i1, i2):
        o1 = self.forward_once(i1)
        o2 = self.forward_once(i2)
        return o1, o2


# 加载预训练的模型并将其移动到指定的设备上
siameseNet = torch.load('siameseNet')
siameseNet = siameseNet.to(device)
siameseNet.eval()


# 获取文件夹下所有文件的路径和对应的标签
def getFn_Dir(tpath):
    dirs = os.listdir(tpath)
    file_labels = []
    for i, dir in enumerate(dirs):
        label = i
        path2 = os.path.join(tpath, dir)
        files = os.listdir(path2)
        for file in files:
            fn = os.path.join(path2, file)
            t = (fn, label)
            file_labels.append(t)
    random.shuffle(file_labels)
    random.shuffle(file_labels)
    random.shuffle(file_labels)

    print("file_labels:")
    print(file_labels[0])
    # print(file_labels[1])
    # print(file_labels[-1])

    return file_labels


# 定义图像处理函数，将图像转换为张量并进行预设的图像处理
def getImg(fn):
    img = Image.open(fn)
    img2 = img.convert('RGB')
    img = np.array(img)
    img = torch.Tensor(img)
    img = transform(img)
    return img


# 定义图像处理函数，将图像转换为RGB格式
def getImg_show(fn):
    img = Image.open(fn)
    img = img.convert('RGB')
    img = np.array(img)
    return img


path = r'E:\Project\Python_Proj\DL\data\faces\testing'

# 获取测试路径下所有图片的路径和标签
fn_labels = getFn_Dir(path)

# 预处理所有的图像，将图像路径和对应的预处理后的图像数据保存在字典中
all_images = {fn: getImg(fn).unsqueeze(0).to(device) for fn, _ in fn_labels}

# 初始化正确预测的数量为 0
correct = 0

# 遍历所有的测试图片
# for fn, label in fn_labels:
#     # 获取当前图片的预处理后的数据
#     img = all_images[fn]
#     # 初始化最小距离和对应的图片、标签和路径
#     img_min, dist_min, label_min, fn_min = -1, 1000, -1, -1
#     # 再次遍历所有的测试图片
#     for fn2, label2 in fn_labels:
#         # 如果是同一张图片，跳过
#         if fn == fn2:
#             continue
#         # 获取另一张图片的预处理后的数据
#         img2 = all_images[fn2]
#         # 使用 Siamese 网络计算两张图片的特征
#         pre_o1, pre_o2 = siameseNet(img, img2)
#         # 计算两张图片特征的距离
#         dist = torch.pairwise_distance(pre_o1, pre_o2, keepdim=True)
#         # 如果当前距离小于最小距离，更新最小距离和对应的图片、标签和路径
#         if dist_min > dist.item():
#             dist_min = dist.item()
#             img_min = img2
#             label_min = label2
#             fn_min = fn2
#     # 如果当前图片的标签和最近图片的标签相同，说明预测正确，正确数量加一
#     correct += int(label == label_min)
#
#     # 获取显示用的图片数据
#     img_show = getImg_show(fn)
#     img_show2 = getImg_show(fn_min)
#
#     # 保存显示用的图片数据
#     images = {fn: img_show, fn_min: img_show2}
#     # 计算相似度
#     stitle = 'Similarity: %.2f' % (dist_min)
#     # 显示两张图片
#     # showTwoImages(images, stitle, 1, 2)
#
# # 输出测试结果，包括测试的图片数量和准确率
# print('一共测试了{:.0f}张图片，准确率为{:.1f}%' \
#       .format(len(fn_labels), 100. * correct / len(fn_labels)))


import cv2
import logging

# 创建一个日志器
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 存储人脸特征的字典
face_features = {}

# 读取人脸文件夹下的所有图片
for fn, label in getFn_Dir('faces'):
    img = getImg(fn).unsqueeze(0).to(device)
    # 使用 Siamese 网络计算图片的特征
    feature = siameseNet.forward_once(img)
    # 存储特征
    face_features[label] = feature

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 从摄像头读取一帧
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像转换为灰度图像，因为getImg函数需要灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用 Siamese 网络计算图像的特征
    feature = siameseNet.forward_once(getImg(gray).unsqueeze(0).to(device))

    # 计算和所有人脸特征的距离
    distances = {label: torch.pairwise_distance(feature, face_feature, keepdim=True).item()
                 for label, face_feature in face_features.items()}

    # 找到距离最小的人脸
    min_label, min_distance = min(distances.items(), key=lambda item: item[1])

    # 如果距离小于某个阈值，认为是这个人
    if min_distance < threshold:
        logger.info(f'识别出 {min_label}')
    else:
        logger.info('陌生人')

    # 显示图像
    cv2.imshow('frame', frame)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()
