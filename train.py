"""
YOLO11-seg 裂缝分割模型训练脚本。

支持两种模式:

【本地 Sanity Check】(快速验证 pipeline)
    python train.py --epochs 5 --batch 8 --model yolo11n-seg.pt --name sanity_check

【完整训练】(在 Kaggle T4 或更强 GPU 上)
    python train.py --epochs 50 --batch 32 --model yolo11s-seg.pt --name full_run

详细的两次实验记录请见 TRAINING_LOG.md
"""
import argparse
import torch
from ultralytics import YOLO


def detect_device(prefer: str = "auto") -> str:
    """自动选择训练设备 (cuda > mps > cpu)。"""
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="YOLO11-seg 裂缝分割训练")
    parser.add_argument("--data", type=str, default="crack.yaml",
                        help="数据集配置文件路径 (默认: crack.yaml)")
    parser.add_argument("--model", type=str, default="yolo11s-seg.pt",
                        help="预训练权重 (默认: yolo11s-seg.pt; sanity 可用 yolo11n-seg.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数 (默认: 50)")
    parser.add_argument("--batch", type=int, default=32,
                        help="batch size (默认: 32; Mac/低显存改 8)")
    parser.add_argument("--imgsz", type=int, default=448,
                        help="输入图像大小 (默认: 448, 与数据集分辨率一致)")
    parser.add_argument("--device", type=str, default="auto",
                        help="训练设备: auto / cuda / mps / cpu (默认: auto)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader 进程数 (默认: 4)")
    parser.add_argument("--project", type=str, default="runs/crack",
                        help="输出目录 (默认: runs/crack)")
    parser.add_argument("--name", type=str, default="full_run",
                        help="实验名 (默认: full_run)")
    parser.add_argument("--patience", type=int, default=20,
                        help="早停 patience (默认: 20)")
    args = parser.parse_args()

    device = detect_device(args.device)
    print(f"🔧 训练配置:")
    print(f"  模型:     {args.model}")
    print(f"  数据集:   {args.data}")
    print(f"  epochs:   {args.epochs}")
    print(f"  batch:    {args.batch}")
    print(f"  imgsz:    {args.imgsz}")
    print(f"  device:   {device}")
    print(f"  output:   {args.project}/{args.name}")
    print()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save=True,
        plots=True,
    )

    print(f"\n✅ 训练完成！结果保存在 {args.project}/{args.name}/")
    print(f"   - best.pt: 最优权重")
    print(f"   - results.png: 训练曲线")


if __name__ == "__main__":
    main()
