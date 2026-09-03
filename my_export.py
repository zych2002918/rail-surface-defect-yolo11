from ultralytics import YOLO

# 加载你的魔改 pt 模型
model = YOLO(r'F:\dalunwen\xiaolunwen\ultralytics-yolo11-main\runs\train\LSDECD-FDPN-ODConv\weights\best.pt')

# 导出模型 (确保 imgsz=640, batch=1, opset=12)
model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    opset=12,
    simplify=True,   # 因为源码改好了，这次可以尝试直接开启简化
    dynamic=False
)