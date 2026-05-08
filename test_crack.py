from ultralytics import YOLO
import os

model = YOLO("yolo11n.pt")

# 批量测试
for img in os.listdir("test_images"):
    results = model(f"test_images/{img}")
    results[0].save(f"results_{img}")
    print(f"{img} 识别到: {results[0].boxes.cls.tolist() if results[0].boxes else '什么都没'}")