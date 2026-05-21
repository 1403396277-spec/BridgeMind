# 训练日志 / Training Log

## Run #1 — Sanity Check (2026-05-08)

**目标**: 验证训练 pipeline 是否正常工作

**配置**:
- 模型: YOLO11n-seg (预训练权重)
- 数据集: Crack Segmentation Dataset (11K+ 多源融合)
- Epochs: 5
- Batch size: 8
- Image size: 448×448
- 设备: Apple M1 Pro (MPS 加速)
- 训练时长: ~3 小时

**结果**:

| Metric | Epoch 1 | Epoch 5 | 提升 |
|--------|---------|---------|------|
| Box mAP@0.5 | 0.127 | **0.468** | +268% |
| Mask mAP@0.5 | 0.069 | **0.359** | +420% |
| Box Precision | 0.324 | 0.673 | +108% |
| Mask Precision | 0.233 | 0.642 | +175% |

**观察**:
- 所有 train/val loss 持续下降，未见过拟合
- 所有 metrics 在 epoch 5 仍处于陡峭上升期，**模型未收敛**
- 验证集预测可视化质量较好，置信度集中在 0.5–0.9
- Confusion matrix 显示主要瓶颈在 recall（漏检 1561 例）

**下一步**:
- [ ] 迁移至 Kaggle T4 GPU 进行 50 epoch 完整训练
- [ ] 探索 yolo11s/yolo11m 更大模型对比
- [ ] 引入数据增强（mosaic, mixup）提升 recall

---

## Run #2 — Full Training on Kaggle (2026-XX-XX)

**目标**: 完整训练，达到工程实用水平

**配置**:
- 模型: YOLO11s-seg (从 yolo11n-seg 升级)
- 数据集: Crack Segmentation Dataset (11K+ 多源融合)
- Epochs: 50
- Batch size: 32
- Image size: 448×448
- 设备: Kaggle Tesla T4 GPU (15GB)
- 优化器: AdamW + Cosine LR
- 数据增强: Mosaic + Mixup + HSV
- 训练时长: ~5 小时

**结果**:

| Metric | Run #1 (5ep, M1) | Run #2 (50ep, T4) | 提升 |
|--------|-------------------|---------------------|------|
| Box mAP@0.5 | 0.468 | **0.688** | +47% |
| Box mAP@0.5:0.95 | 0.295 | **0.485** | +64% |
| Mask mAP@0.5 | 0.359 | **0.561** | +56% |
| Box Precision | 0.673 | **0.788** | +17% |
| Box Recall | 0.43 | **0.596** | +39% |
| 推理速度 | - | **3.1ms/img** | - |

**结论**: 
- 模型达到工程实用水平（mAP@0.5 ≈ 0.69 接近论文级）
- 推理速度满足实时检测需求（>300 FPS）
- best.pt 模型已保存，进入 pipeline 整合阶段