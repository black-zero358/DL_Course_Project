from time import sleep
from torch import nn
import random
import cv2
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设备选择，如果有可用的CUDA设备则使用CUDA，否则使用CPU

torch.manual_seed(123)
random.seed(123)
np.random.seed(10)
# 设置随机种子，使得实验结果可复现


transform = transforms.Compose(
    [transforms.Resize((100, 100)),
     transforms.Grayscale(num_output_channels=1),  # 添加这一行
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])


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

    return file_labels


# 定义图像处理函数，将图像转换为张量并进行预设的图像处理
def getImg(fn):
    img = Image.open(fn)
    img2 = img.convert('RGB')
    img = np.array(img)
    img = torch.Tensor(img)
    img = transform(img)
    return img


# 这里是你的模型和预处理操作
# ...

# 存储人脸特征的字典
face_features = {}

# 计算并存储每个人脸图片的特征
face_folder = r'.\faces'  # 你的人脸图片文件夹
for face_name in os.listdir(face_folder):
    face_img = Image.open(os.path.join(face_folder, face_name))
    face_img = transform(face_img).unsqueeze(0).to(device)
    face_feature = siameseNet.forward_once(face_img).detach().cpu().numpy()
    face_features[face_name] = face_feature

print("已录入信息")
print(face_features)

# 打开摄像头
cap = cv2.VideoCapture(0)
print('打开摄像头')

while True:
    # 读取摄像头的帧
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧进行预处理，并通过模型得到特征
    frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_img = transform(frame_img).unsqueeze(0).to(device)
    frame_feature = siameseNet.forward_once(frame_img).detach().cpu().numpy()

    # 计算这个特征和所有存储的特征的距离
    distances = {name: np.linalg.norm(feature - frame_feature) for name, feature in face_features.items()}

    # 找出最小的距离
    min_distance_name, min_distance = min(distances.items(), key=lambda x: x[1])

    threshold = 1.5

    # 如果最小的距离小于某个阈值，那么就认为这个人是对应的人，否则就是陌生人
    if min_distance < threshold:
        cv2.putText(frame, min_distance_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # print(min_distance_name)
    else:
        cv2.putText(frame, f"Unknown distance:{min_distance}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)



    # 显示帧
    cv2.imshow('Frame', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    sleep(1)

# 释放摄像头
cap.release()
cv2.destroyAllWindows()
