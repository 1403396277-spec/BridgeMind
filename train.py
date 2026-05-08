from ultralytics import YOLO
import torch

# 检查 MPS（M 系列芯片的GPU加速）
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载分割模型（注意是 -seg 后缀）
model = YOLO("yolo11n-seg.pt")  # 第一次会自动下载

# 训练
results = model.train(
    data="crack.yaml",      # 你之前创建的配置文件
    epochs=5,                # 先只跑5个 epoch 验证
    imgsz=448,              # 数据集本来就是 448x448
    batch=8,                # Mac 内存有限，先小批量
    device=device,
    workers=4,
    project="runs/crack",
    name="sanity_check",
    patience=20,
    save=True,
    plots=True,
)

print("\n训练完成！结果保存在 runs/crack/sanity_check/")