# 导入Ultralytics的YOLO模块
from ultralytics import YOLO

# 加载一个模型
model = YOLO(r'runs\detect\train5\weights\best.onnx')  # 预训练的YOLOv8n模型

# 对一系列图片进行批量推理
results = model(['test.jpg'])

# 处理结果列表
for result in results:
    boxes = result.boxes  # 边界框对象，用于输出边界框
    masks = result.masks  # 掩膜对象，用于输出分割掩膜
    keypoints = result.keypoints  # 关键点对象，用于输出姿态
    probs = result.probs  # 概率对象，用于输出分类结果
    result.show()  # 显示
    result.save(filename='result.jpg')  # 保存
