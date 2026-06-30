---
type: learning-path
direction: efficient-training-inference
maturity: draft
updated: 2026-06-30
---

# 高效训练与推理（Efficient Training & Inference）学习路线

## 0. 这个方向在解决什么
结论先行：这个方向关心「在不显著掉点的前提下，让大模型训得更快、推得更省」。手段包括量化、蒸馏、稀疏/缓存、长上下文优化、采样步数压缩、系统级 kernel 优化等。仓库内目前样本少，集中在生成式世界模型的推理/训练加速：X-Cache 用 cross-chunk 残差缓存给 few-step AR DiT 提速，X-Foresight 用 block-sparse attention 降训练 step time。因此本路线把仓库内两篇作为「真实案例」，前置与系统方法主要链外部资源。

## 1. 前置知识
- 推理加速通用法：KV cache、投机解码、量化（INT8/4、AWQ/GPTQ）、蒸馏、剪枝。
- 注意力优化：FlashAttention 系列、block-sparse attention，理解 step time 来自哪。
- 扩散/DiT 专属：采样步数压缩（few-step、consistency/AR）、特征/残差缓存复用（cross-step/cross-chunk caching）。
- 评测：wall-clock speedup、step time、以及加速后是否保 PSNR/SSIM/LPIPS 等质量指标。
- 背景：相关世界模型见 [world-models 学习路线](./world-models.md)。

## 2. 奠基论文（仓库内）
- ★ [X-Cache](../papers/efficient-training-inference/2026-x-cache.md) — 针对 X-World few-step AR DiT 推理，用 cross-chunk block residual cache + action-aware fingerprint + KV-update protection，获得约 2.6-2.7x DiT wall-clock 加速且基本保质量，是仓库内推理加速的核心样本。
- [X-Foresight](../papers/world-models/2026-x-foresight.md) — 报告 block-sparse attention（带 mask）相对 FlashAttention-2 把训练 step time 从 24.50s 降到 15.40s，是训练侧效率的样本。

## 3. 近期 SOTA 与分支
- [XPeng X 系列自动驾驶世界模型横向对比](../comparisons/world-models/xpeng-x-series-world-models.md) — 仓库内唯一相关对比报告，X-Cache（推理）与 X-Foresight（训练）的效率定位在此串联。
- 分支轴：按阶段分「训练加速（稀疏注意力、并行/混合精度）」vs「推理加速（缓存复用、步数压缩、量化）」；按是否改权重分「training-free（X-Cache 这类缓存）」vs「需重训（蒸馏/量化感知训练）」。
- 提示：仓库样本偏生成式世界模型，要补 LLM 侧（KV cache/投机解码/量化）与系统侧 kernel 优化需链外部综述与官方文档。

## 4. 动手：最小复现路径
- 最小可跑路径：本方向仓库内两篇均缺公开代码/权重/数据，无法直接复现。建议先用外部开源栈练手——如对一个小 DiT/LLM 做 INT8 量化或 KV-cache profiling，建立 wall-clock vs 质量的测量直觉，再回看 X-Cache 的缓存设计。
- 复现记录：planned，仓库暂无复现记录。X-Cache / X-Foresight 缺公开代码、权重和内部数据，复现 blocked。

## 5. 常见误区 / 我的判断
- 误区：只报 speedup 不报质量。任何加速都要同时给出 PSNR/SSIM/LPIPS 或任务指标，否则无法判断是否「免费午餐」。
- 误区：把内部数据上的 step time/wall-clock 当通用结论。X-Cache/X-Foresight 的数字依赖其特定模型与数据，难外推。
- 判断：仓库内样本少且 blocked，本方向更适合「读这两篇理解缓存/稀疏注意力设计 + 用外部开源栈做量化/缓存实操」并行推进；等代码 release 再补复现。
