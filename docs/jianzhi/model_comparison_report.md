 # 钢轨划痕检测模型性能对比报告

> 生成时间：2026-03-21  
> 任务：钢轨表面划痕目标检测（单类别）  
> 输入分辨率：640×640  
> 优化器：SGD，lr₀=0.01，weight_decay=0.0005

---

## 1. 模型概览

**表 1.1 各模型基本信息概览**

*Table 1.1 Overview of model configurations*

| 模型 / Model | 类型 | 训练轮次 / Epochs | Batch Size | Parameters / M | GFLOPs / G |
|---|---|:---:|:---:|:---:|:---:|
| **yolo11s** | 基线模型（官方YOLOv11s） | 401 | 16 | 9.429 | 21.55 |
| **yolo11s-rail-pruned-conservative** | yolo11s 保守剪枝 | 300 | 128 | 8.549 | 18.19 |
| **LSDECD-FDPN-ODConv** | 改进模型（LSDECD + FDPN + ODConv） | 500 | 16 | 10.164 | 19.26 |
| **LSDECD-FDPN-ODConv-pruned-v2** | 改进模型 + 通道剪枝 v2 | 500 | 128 | 9.345 | 16.58 |

> **关系说明**：`LSDECD-FDPN-ODConv-pruned-v2` 是对 `LSDECD-FDPN-ODConv` 进行通道降维剪枝后得到的压缩版本；`yolo11s-rail-pruned-conservative` 是对 `yolo11s` 进行保守剪枝后得到的压缩版本。

### 模型结构说明

- **yolo11s**：官方标准 YOLOv11s，使用 `yolo11.yaml`，作为性能基线参照。
- **yolo11s-rail-pruned-conservative**：在 yolo11s 基础上进行保守通道剪枝，降低参数量与计算量，但精度有所下降，属于压缩参考对比。
- **LSDECD-FDPN-ODConv**：完整改进版模型，集成三项主要创新：
  - **LSDECD 检测头**：含 `Detect_LSDECD` 结构，引入 `share_conv`（DEConv_GN 可形变卷积），针对细长划痕特征强化局部几何感知。
  - **FDPN（Feature Dimension Pyramid Network）**：改进特征金字塔网络，增强多尺度特征融合能力。
  - **ODConv（Omni-Dimensional Dynamic Convolution）**：全维度动态卷积，从滤波器空间/通道/深度多维度自适应调整卷积权重。
- **LSDECD-FDPN-ODConv-pruned-v2**：在完整改进模型基础上进行通道降维剪枝。保持 `scales: s:[0.50, 0.50, 1024]` 与原始一致，通过在 yaml 中写 2 倍目标值的方式将骨干 P3/P4 通道缩减约 12.5%，P5/SPPF/C2PSA 完全保留，`Detect_LSDECD` 的 `hidc` 维持原始值 128，实现真正有效的压缩。

---

## 2. 检测性能对比（最佳 Epoch）

> 以验证集上 mAP50 最高的 Epoch 统计各项指标。

**表 2.1 各模型检测性能对比（最佳 Epoch）**

*Table 2.1 Detection performance comparison of each model (best epoch)*

| 模型 / Model | 最佳 Epoch | Precision / % | Recall / % | mAP@0.5 / % | mAP@0.5:0.95 / % |
|---|:---:|:---:|:---:|:---:|:---:|
| yolo11s | 395 / 401 | 85.29 | 84.67 | 86.91 | 65.93 |
| yolo11s-rail-pruned-conservative | 206 / 300 | 86.44 | 83.81 | 85.29 | 61.75 |
| LSDECD-FDPN-ODConv | 463 / 500 | 87.91 | **87.70** | **88.95** | **67.35** |
| LSDECD-FDPN-ODConv-pruned-v2 | 426 / 500 | **87.66** | 85.12 | 88.16 | 66.39 |

### 2.1 相对基线（yolo11s）变化量

**表 2.2 各模型相对基线（yolo11s）变化量**

*Table 2.2 Performance and complexity changes relative to baseline (yolo11s)*

| 模型 / Model | ΔPrecision | ΔRecall | ΔmAP@0.5 | ΔmAP@0.5:0.95 | ΔParameters / M | ΔGFLOPs / G |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| yolo11s-rail-pruned-conservative | +1.35% | -1.02% | **-1.86%** | -6.34% | -0.880 (↓**9.32%**) | -3.36 (↓**15.59%**) |
| LSDECD-FDPN-ODConv | +3.07% | +3.57% | **+2.35%** | +2.15% | +0.735 (↑**7.80%**) | -2.29 (↓**10.63%**) |
| LSDECD-FDPN-ODConv-pruned-v2 | +2.78% | +0.53% | **+1.44%** | +0.70% | -0.084 (↓**0.89%**) | -4.97 (↓**23.06%**) |

### 2.2 剪枝效益专项分析

> 本节分别对两组"剪枝前 vs 剪枝后"模型进行直接对比，量化剪枝带来的结构压缩收益与精度代价。

#### 组一：LSDECD-FDPN-ODConv → LSDECD-FDPN-ODConv-pruned-v2（改进模型剪枝）

**表 2.3 改进模型剪枝前后对比（LSDECD-FDPN-ODConv）**

*Table 2.3 Comparison before and after pruning of the improved model*

| 指标 / Metric | 未剪枝 / LSDECD-FDPN-ODConv | 剪枝后 / pruned-v2 | 变化量 | 变化率 |
|---|:---:|:---:|:---:|:---:|
| Parameters / M | 10.164 | 9.345 | -0.819 | ↓ **8.06%** |
| GFLOPs / G | 19.26 | 16.58 | -2.68 | ↓ **13.91%** |
| Precision / % | 87.91 | 87.66 | -0.25 | ↓ 0.28% |
| Recall / % | 87.70 | 85.12 | -2.58 | ↓ **2.94%** |
| **mAP@0.5 / %** | **88.95** | **88.16** | **-0.79** | **↓ 0.89%** |
| mAP@0.5:0.95 / % | 67.35 | 66.39 | -0.96 | ↓ 1.43% |

> **结论**：剪枝后参数量减少 **8.06%**，计算量减少 **13.91%**，而 mAP50 仅下降 **0.89%**，精度损失极小，剪枝性价比极高。Recall 下降 2.94% 是主要代价，意味着漏检率略有提升，需根据场景权衡。

---

#### 组二：yolo11s → yolo11s-rail-pruned-conservative（基线模型剪枝）

**表 2.4 基线模型剪枝前后对比（yolo11s）**

*Table 2.4 Comparison before and after pruning of the baseline model*

| 指标 / Metric | 未剪枝 / yolo11s | 剪枝后 / pruned-conservative | 变化量 | 变化率 |
|---|:---:|:---:|:---:|:---:|
| Parameters / M | 9.429 | 8.549 | -0.880 | ↓ **9.33%** |
| GFLOPs / G | 21.55 | 18.19 | -3.36 | ↓ **15.59%** |
| Precision / % | 85.29 | 86.44 | +1.15 | ↑ 1.35% |
| Recall / % | 84.67 | 83.81 | -0.86 | ↓ 1.02% |
| **mAP@0.5 / %** | **86.91** | **85.29** | **-1.62** | **↓ 1.86%** |
| mAP@0.5:0.95 / % | 65.93 | 61.75 | -4.18 | ↓ **6.34%** |

> **结论**：剪枝后参数量减少 **9.33%**，计算量减少 **15.59%**，但 mAP50 下降 **1.86%**，mAP50-95 大幅下降 **6.34%**，精度损失不可忽视。Precision 略有提升但 Recall 下降，说明剪枝导致模型趋于保守（漏检增加）。此方案剪枝过于激进，建议精细化调整。

---

#### 组三：综合对比——pruned-v2 相对 yolo11s 基线的收益

**表 2.5 LSDECD-FDPN-ODConv-pruned-v2 与 yolo11s 基线综合对比**

*Table 2.5 Comprehensive comparison between pruned-v2 and yolo11s baseline*

| 指标 / Metric | yolo11s（基线） | LSDECD-FDPN-ODConv-pruned-v2 | 变化量 | 变化率 |
|---|:---:|:---:|:---:|:---:|
| Parameters / M | 9.429 | 9.345 | -0.084 | ↓ 0.89% |
| GFLOPs / G | 21.55 | 16.58 | -4.97 | ↓ **23.06%** ✅ |
| Precision / % | 85.29 | 87.66 | +2.37 | ↑ **2.78%** ✅ |
| Recall / % | 84.67 | 85.12 | +0.45 | ↑ 0.53% ✅ |
| **mAP@0.5 / %** | **86.91** | **88.16** | **+1.25** | ↑ **1.44%** ✅ |
| mAP@0.5:0.95 / % | 65.93 | 66.39 | +0.46 | ↑ 0.70% ✅ |

> **结论**：LSDECD-FDPN-ODConv-pruned-v2 与 yolo11s 参数量相当（仅差 0.084M），但计算量降低 **23.06%**，同时四项精度指标**全面超越**基线，是本次实验综合性能最优的部署候选模型。

---

## 3. 最终 Epoch 指标（训练结束时）

**表 3.1 各模型最终 Epoch 检测指标**

*Table 3.1 Detection metrics at the final training epoch*

| 模型 / Model | Precision / % | Recall / % | mAP@0.5 / % | mAP@0.5:0.95 / % |
|---|:---:|:---:|:---:|:---:|
| yolo11s | 85.20 | 84.69 | 86.87 | 65.85 |
| yolo11s-rail-pruned-conservative | 83.95 | 83.94 | 84.17 | 62.16 |
| LSDECD-FDPN-ODConv | 87.79 | 87.21 | 88.92 | 67.31 |
| LSDECD-FDPN-ODConv-pruned-v2 | 88.42 | 85.50 | 87.90 | 67.06 |

---

## 4. 验证集损失对比（最佳 Epoch 时）

**表 4.1 各模型验证集损失对比（最佳 Epoch）**

*Table 4.1 Validation loss comparison at best epoch*

| 模型 / Model | val/box_loss | val/cls_loss | val/dfl_loss |
|---|:---:|:---:|:---:|
| yolo11s | 1.47604 | 0.72945 | 0.96168 |
| yolo11s-rail-pruned-conservative | 1.65591 | 0.87069 | 0.99980 |
| LSDECD-FDPN-ODConv | 1.51179 | 0.76432 | 0.91650 |
| LSDECD-FDPN-ODConv-pruned-v2 | 1.54496 | 0.77790 | 0.92677 |

> 注：yolo11s 的 box_loss 最低，反映其定位回归偏差小；LSDECD 系列的 dfl_loss 更低，说明分布焦点损失收敛更好，边界框分布拟合能力更优。

---

## 5. 训练收敛过程（mAP50 @ 关键 Epoch）

**表 5.1 各模型关键 Epoch 处的 mAP@0.5 收敛情况**

*Table 5.1 mAP@0.5 convergence at key epochs*

| Epoch | yolo11s | yolo11s-pruned | LSDECD-FDPN-ODConv | LSDECD-FDPN-ODConv-pruned-v2 |
|:---:|:---:|:---:|:---:|:---:|
| 50 | 0.8133 | 0.7911 | **0.8194** | 0.8020 |
| 100 | 0.8476 | 0.8162 | **0.8571** | 0.8480 |
| 200 | 0.8590 | 0.8409 | **0.8708** | 0.8653 |
| 300 | 0.8681 | 0.8417 | **0.8834** | 0.8702 |
| 最终最佳 | 0.8691 | 0.8529 | **0.8895** | 0.8816 |

**收敛分析**：
- LSDECD-FDPN-ODConv 全程领先，在 epoch 50 时已达 0.8194，收敛最快。
- LSDECD-FDPN-ODConv-pruned-v2 与 yolo11s 早期相当，后期稳定超越基线。
- yolo11s-rail-pruned-conservative 全程最低，且在 epoch 200 后趋于平台，精度提升潜力有限，反映过度保守剪枝损伤了模型容量。

---

## 6. 综合效率分析

### 精度-计算量权衡（mAP50 vs GFLOPs）

```
mAP@0.5
88.95% |          ★ LSDECD-FDPN-ODConv (19.26G)
88.16% |    ★ LSDECD-FDPN-ODConv-pruned-v2 (16.58G)  ← 最优效率点
86.91% |                    ★ yolo11s (21.55G)
85.29% |  ★ yolo11s-rail-pruned-conservative (18.19G)
       +------+------+------+------+------→ GFLOPs
       16     17     18     19     20     21
```

**表 6.1 模型精度-效率综合对比**

*Table 6.1 Comprehensive accuracy-efficiency comparison*

| 模型 / Model | Parameters / M | GFLOPs / G | mAP@0.5 / % | mAP@0.5 / GFLOPs | 说明 |
|---|:---:|:---:|:---:|:---:|---|
| yolo11s | 9.429 | 21.55 | 86.91 | 0.04035 | 基线 |
| yolo11s-rail-pruned-conservative | 8.549 | 18.19 | 85.29 | 0.04689 | 效率比高但绝对精度低 |
| LSDECD-FDPN-ODConv | 10.164 | 19.26 | 88.95 | 0.04619 | 精度最高，效率较优 |
| **LSDECD-FDPN-ODConv-pruned-v2** | **9.345** | **16.58** | **88.16** | **0.05317** | **最高效率比，精度超基线 ✅** |

---

## 7. 结论与建议

### 总结

| 维度 | 推荐模型 | 理由 |
|---|---|---|
| **最高精度** | LSDECD-FDPN-ODConv | mAP@0.5=88.95%，Recall=87.70%，全面领先 |
| **最佳效率（精度/算力）** | **LSDECD-FDPN-ODConv-pruned-v2** | GFLOPs 最低（16.58G），mAP50 仍超基线 1.44% |
| **最轻量** | yolo11s-rail-pruned-conservative | 参数最少（8.549M），但精度低于基线 |
| **基线参照** | yolo11s | 官方标准模型，mAP@0.5=86.91% |

### 核心结论

1. **LSDECD-FDPN-ODConv** 是本次实验精度最高的模型，三项结构改进协同发挥效果：LSDECD 检测头增强了细长划痕的局部形变感知，FDPN 改善了多尺度融合，ODConv 提升了特征表达能力。
   - 相比 yolo11s 基线：mAP50 提升 **+2.35%**（86.91%→88.95%），Recall 提升 **+3.57%**（漏检率明显降低），GFLOPs 降低 **10.63%**（21.55G→19.26G）
   - 代价：参数量增加 **7.80%**（9.429M→10.164M）

2. **LSDECD-FDPN-ODConv-pruned-v2** 是推荐部署模型，剪枝效益分析：
   - **相比未剪枝版本（LSDECD-FDPN-ODConv）**：参数量减少 **8.06%**（10.164M→9.345M），GFLOPs 降低 **13.91%**（19.26G→16.58G），mAP50 仅下降 **0.89%**（88.95%→88.16%），剪枝性价比极高
   - **相比 yolo11s 基线**：参数量几乎持平（↓0.89%），GFLOPs 大幅降低 **23.06%**（21.55G→16.58G），同时 mAP50 反而提升 **+1.44%**，Precision 提升 **+2.78%**，四项精度指标全面超越基线

3. **yolo11s-rail-pruned-conservative** 剪枝代价分析：
   - 相比 yolo11s：参数量减少 **9.33%**，GFLOPs 降低 **15.59%**
   - 但 mAP50 下降 **1.86%**，mAP50-95 大幅下降 **6.34%**，精度损失代价过高
   - 最佳 Epoch 仅第 206 轮（训练提前饱和），说明剪枝过激损伤了模型容量，建议精细化调整剪枝比例

4. 从收敛速度来看，LSDECD-FDPN-ODConv 系列在训练前期（epoch 50）即达 81.94%（基线仅 81.33%），说明改进结构有助于更快收敛，在有限训练预算下同样具有优势。

### 部署建议

- **在线实时检测（边缘端/低算力）**：选用 `LSDECD-FDPN-ODConv-pruned-v2`，GFLOPs 16.58G，精度高于基线。
- **离线高精度检测（服务器端）**：选用 `LSDECD-FDPN-ODConv`，mAP@0.5=88.95%，Recall 最高（漏检最少）。
- **快速验证/对照实验**：使用 `yolo11s` 基线。
- **不建议** 直接部署 `yolo11s-rail-pruned-conservative`，需进一步优化剪枝策略后再评估。

---

## 附录：原始训练配置

**表 A.1 各模型原始训练超参数**

*Table A.1 Original training hyperparameters for each model*

| 参数 / Param | LSDECD-FDPN-ODConv-pruned-v2 | LSDECD-FDPN-ODConv | yolo11s | yolo11s-rail-pruned-conservative |
|---|---|---|---|---|
| model yaml | LSDECD-FDPN-ODConv-pruned-v2.yaml | LSDECD-FDPN-ODConv.yaml | yolo11.yaml | yolo11s-rail-pruned-conservative.yaml |
| epochs | 500 | 500 | 500（停 401） | 500（停 300） |
| batch | 128 | 16 | 16 | 128 |
| imgsz | 640 | 640 | 640 | 640 |
| optimizer | SGD | SGD | SGD | SGD |
| lr0 | 0.01 | 0.01 | 0.01 | 0.01 |
| weight_decay | 0.0005 | 0.0005 | 0.0005 | 0.0005 |
