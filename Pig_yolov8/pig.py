# 导入Ultralytics的YOLO模块
from ultralytics import YOLO

def main():
    # 加载一个预训练的YOLO模型
    model = YOLO('yolov8n.pt')
    # 使用'coco8.yaml'数据集训练模型，训练3个epoch，workers=0表示不使用额外的进程进行数据加载
    result = model.train(data='E:\Project\Python_Proj\DL\proj\Pig_yolov8\data.yaml', epochs=3, workers=0)
    # 在验证集上评估模型的性能
    results = model.val()
    # 将模型导出为ONNX格式
    success = model.export(format='onnx')

if __name__ == '__main__':
    main()
