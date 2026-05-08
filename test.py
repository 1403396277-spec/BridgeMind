from ultralytics import YOLO

# 第一次运行会自动下载模型，等几分钟
model = YOLO("yolo11n.pt")

# 用官方测试图
results = model("https://ultralytics.com/images/bus.jpg")

# 显示结果
results[0].show()