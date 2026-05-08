# BridgeMind 🌉

> 基于多模态大模型的桥梁缺陷智能诊断与规范咨询系统

## 项目简介

BridgeMind 是一个端到端的桥梁缺陷智能诊断系统，结合计算机视觉与大语言模型技术，实现从图像输入到诊断报告输出的全自动流程。

## 技术栈

- **检测/分割模型**：YOLO11-seg
- **数据集**：Crack Segmentation Dataset (11K+ images, 多源融合)
- **大语言模型**：DeepSeek-V3 API
- **检索增强**：LlamaIndex + BGE-M3 Embedding
- **前端展示**：Gradio

## 项目状态

🚧 开发中

- [x] 数据集准备与预处理
- [x] YOLO11-seg 模型训练
- [ ] 桥梁规范 RAG 系统
- [ ] LLM 诊断报告生成
- [ ] Gradio Web 界面
- [ ] HuggingFace Space 部署

## 作者

党成慧泽 (DANG Chenghuize)