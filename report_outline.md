# Image2LoRA Report 大纲

## 写作主线

本报告建议以 **style information** 为核心组织，而不是单纯介绍一个 Image2LoRA 系统。主线可以写成：

> 我们关心的问题不是“如何生成一张好看的图”，而是“参考图中的 style 信息能否被提取出来、压缩成模型参数、注入扩散模型，并在生成结果中被稳定表达出来”。

因此整篇 report 可以围绕四个层次展开：style 信息是什么，如何从参考图中提取 style 信息，如何把 style 信息转化为 LightLoRA 参数，如何证明生成结果确实利用了这些 style 信息。

## Suggested Title

**Learning Style Information as Dynamic LightLoRA Parameters**

可选副标题：

**A Single-Reference Style Transfer Framework Based on Visual Context and HyperNetwork**

## Report Outline

### 1. Introduction

从 style 信息的角度定义任务：给定一张参考图和一段文本提示词，模型需要生成一张内容符合文本、视觉风格接近参考图的图像。这里的关键不是复现参考图的具体物体，而是迁移其中更抽象的 style information，例如色调、纹理、材质、光照、构图倾向和绘画感。

本节重点说明本项目的核心问题：单张参考图中包含足够的 style 上下文吗？这些 style 信息能否被一个统一模型提取并转化为可复用的生成控制信号？相比为每个风格单独训练 DreamBooth 或 LoRA，我们希望学习一个从参考图到风格参数的通用映射。

### 2. What is Style Information?

本节专门讨论 style information 的含义。可以把 style 拆成几个层面：低层视觉统计，如颜色分布、对比度和局部纹理；中层视觉模式，如材质、笔触、边缘形态和光照；高层审美倾向，如画面氛围、艺术化程度和抽象表达方式。

同时说明 style 与 content 的区别。文本提示词主要约束图像内容，例如山、河流、宫殿或人物；参考图则提供 style 信息，影响生成图像的外观分布。我们的工作重点是让模型尽量保留文本语义，同时从参考图中借用 style。

### 3. Motivation and Related Work

围绕 style 信息的建模方式介绍相关工作。DreamBooth 和传统 LoRA 可以把某个特定风格写入模型参数，但它们通常需要针对每个风格单独优化。HyperDreamBooth 说明图像信息可以被 HyperNetwork 转化为参数更新。Lightweight DreamBooth 进一步启发我们：不一定要预测完整 LoRA 矩阵，可以先构造一个较小的参数子空间。

本节的落点是：已有方法证明 style 可以被参数化，但逐风格训练不够灵活；我们的方向是学习一个通用 style encoder-to-parameter mapping，让参考图中的 style 信息在一次前向传播中变成动态 LightLoRA。

### 4. Style Representation from Reference Image

介绍如何表示参考图中的 style 信息。本项目使用冻结的 DINOv2 作为图像编码器，从参考图中提取 patch-level visual tokens。相比只用全局图像 embedding，patch token 保留了更细粒度的视觉模式，有助于捕捉局部纹理、颜色区域、材质变化和笔触结构。

这里可以强调一个假设：DINOv2 特征虽然常用于语义表征，但其 patch-level 特征仍包含可用于 style transfer 的视觉上下文。我们不是直接对参考图做像素级匹配，而是把参考图编码成一组视觉 token，再交给 HyperNetwork 学习哪些信息应该转化为生成模型参数。

### 5. Style-to-Parameter Mapping

本节是方法核心：如何把 style representation 转成模型可执行的控制信号。DINOv2 输出的 visual tokens 被输入到 transformer decoder 结构的 HyperNetwork 中。HyperNetwork 通过交叉注意力读取参考图特征，并经过多次 iterative refinement，生成每个 UNet 注入层对应的 LightLoRA embedding。

可以把这个过程写成从“视觉上下文”到“参数上下文”的转换：参考图中的 style 信息并不直接作为额外图像条件输入扩散模型，而是被转化为扩散模型内部 attention linear layers 的动态低秩调制参数。这使得 style 控制发生在模型参数层面，而不是单纯依赖 prompt 或图像拼接。

推荐方法框图：

```text
Reference Image
    -> DINOv2 Patch Tokens
    -> HyperNetwork
    -> LightLoRA Embeddings
    -> Dynamic UNet Attention Modulation
    -> Stylized Generation
```

### 6. LightLoRA as a Compact Style Carrier

解释为什么 LightLoRA 适合作为 style 信息载体。普通 LoRA A/B 矩阵参数量较大，如果让 HyperNetwork 直接预测完整矩阵，会增加学习难度和输出维度。本项目采用辅助矩阵设计：HyperNetwork 只预测较小的 embedding，再与训练得到的 `down_aux` 和 `up_aux` 组合，恢复最终 LoRA 权重。

当前配置中 rank=1、down_dim=64、up_dim=32，因此每层只需要预测 96 个标量。这个设计可以被解释为一种 style bottleneck：模型被迫把参考图中的 style 信息压缩进较小的参数空间。这样既降低了预测难度，也让动态 LoRA 更轻量，适合快速从单张参考图生成风格控制参数。

### 7. Training for Style Injection

从 style 注入角度介绍训练流程。每个样本包含参考风格图、目标图像和 caption。训练时，参考图先被编码为 style representation，再生成动态 LightLoRA；目标图像被编码到 latent space 并加噪；注入 LightLoRA 的冻结 SD1.5 UNet 需要根据文本条件预测噪声。

训练目标仍然是标准扩散噪声预测 MSE，但它间接要求 HyperNetwork 学会：哪些参考图信息能帮助生成目标风格图。训练中 SD1.5、VAE、文本编码器和 DINOv2 基本冻结，主要学习 HyperNetwork 与 LightLoRA 辅助矩阵。因此可以把训练目标概括为：学习一个从 reference style information 到 diffusion parameter modulation 的映射。

### 8. Implementation

结合代码仓库说明 style 信息流如何落地。`DINOv2Encoder` 负责提取参考图 patch tokens；`ImageHyperDream` 和 `ImageWeightGenerator` 负责把 style tokens 解码成 LightLoRA embeddings；`LoRAModule` 用 `down_aux`、`up_aux` 和 embedding 组合出实际 LoRA 权重；`LoRANetwork` 将这些权重注入 SD1.5 UNet 的 attention linear layers。

脚本层面，`scripts/train.py` 实现 style-to-parameter 的训练闭环，`scripts/infer.py` 实现单张参考图到动态 LoRA 再到生成图像的推理过程，`scripts/batch_infer.py` 和 `scripts/evaluate.py` 支持大规模生成和 style 相关指标评估。

### 9. Experiments: Does the Model Use Style Information?

实验部分建议围绕一个判断标准组织：生成结果是否真的利用了参考图中的 style information，而不是只按照文本提示词生成普通图像。

第一组实验使用 `image_in_ppt/` 中的三组可视化案例，对比参考图、baseline SD1.5 输出和 Image2LoRA 输出。观察重点包括颜色是否接近参考图、纹理和材质是否更相似、画面是否更有参考图的艺术化倾向。

第二组实验使用 `outputs/batch_eval/` 的批量结果，覆盖 9 个风格、3744 个生成样本。通过 FID 和 VGG style loss 评估生成图像与参考风格分布之间的距离。

### 10. Results: Evidence of Style Transfer

定性结果可以写：在山景、宫殿和河流等样例中，baseline 更多体现文本语义，输出像普通 SD1.5 图像；Image2LoRA 则更明显地迁移了参考图的色彩倾向、纹理细节、材质质感和绘画感。这说明参考图中的 style 信息通过动态 LightLoRA 改变了生成分布。

定量结果建议使用以下表格：

| Setting | Metric | Baseline | Image2LoRA |
|---|---:|---:|---:|
| 3 visual cases | Gram style loss | 0.00260 | 0.00177 |
| 3 visual cases | AdaIN style loss | 37.57 | 24.04 |

批量实验可以概括为：在 9 个风格、3744 个样本上，平均 FID 约为 341.8，平均 Gram style loss 约为 0.00186，平均 AdaIN style loss 约为 24.0。因为这些指标都是 lower is better，三组展示样例中 style loss 的下降可以作为模型利用 style 信息的初步证据。

### 11. Discussion: Style, Content, and Generalization

本节讨论 style 信息建模中的 trade-off。style transfer 并不是简单让输出图像完全接近参考图，因为还需要保留文本内容。如果 style 控制过强，可能损害语义细节；如果 style 控制过弱，则输出接近 baseline，无法体现参考图。

还可以讨论统一 HyperNetwork 的泛化问题。它比逐风格训练更通用，但也可能在某些单一风格上不如专门微调的 LoRA 精细。因此本项目更适合被定位为一种快速、轻量、通用的 style parameter generation 方法，而不是每个风格上的最优个性化微调方案。

### 12. Limitations

当前评估仍然有限。首先，现有实验缺少真实 content image，因此 SSIM、LPIPS 和 ArtFID 等内容一致性指标无法完整使用。其次，FID 对样本规模和参考集构造比较敏感，不能单独代表风格迁移质量。第三，style 本身包含多层信息，仅用 Gram loss 或 AdaIN loss 难以完整衡量。第四，当前模型仍可能出现风格迁移不足、文本细节偏移或图像质量波动。

### 13. Future Work

后续工作可以继续沿 style 信息主线推进。第一，加入 rank-relaxed finetuning，让动态生成的 LightLoRA 在单张参考图上进一步适配。第二，研究不同 style representation 的影响，例如 DINO、CLIP、VGG 或多尺度特征。第三，系统比较 LoRA rank、注入层范围、辅助矩阵维度和 HyperNetwork 深度对 style 迁移的影响。第四，引入更完整的 style evaluation，包括 CLIP/DINO 特征相似度、CLIPScore、VQA-based score、KID、aesthetic score 和人类偏好实验。

### 14. Conclusion

总结时回到 style information 主线：本项目尝试把单张参考图中的 style 信息看作一种视觉上下文，并学习将其转化为动态 LightLoRA 参数。通过 DINOv2 提取 style representation，通过 HyperNetwork 完成 style-to-parameter mapping，再通过 LightLoRA 注入冻结扩散模型，Image2LoRA 在不逐风格训练 LoRA 的情况下，实现了初步的参考图风格迁移。实验结果说明 style 信息确实可以通过轻量参数调制影响生成分布，也为后续更系统的 style representation 和 style injection 研究提供了基础。

## 推荐图表安排

| 位置 | 图表内容 | 说明 |
|---|---|---|
| Introduction | Style transfer task 示意图 | 展示 reference image、text prompt、stylized output |
| What is Style Information? | style/content 拆分图 | 区分语义内容与色调、纹理、材质等 style 维度 |
| Style Representation | DINOv2 patch tokens 示意图 | 说明参考图如何变成视觉 token |
| Style-to-Parameter Mapping | HyperNetwork 生成 LightLoRA 流程图 | 突出 style 信息从图像特征到参数的转换 |
| LightLoRA | 辅助矩阵压缩图 | 对比完整 LoRA 预测和 compact embedding 预测 |
| Experiments | 三组 reference/baseline/output 对比 | 来自 `image_in_ppt/` |
| Results | style loss 对比表 | 来自 `outputs/eval/metrics_report.json` |

## 写作注意事项

报告中应尽量多使用 “style information”、“style representation”、“style-to-parameter mapping”、“style injection”、“style carrier”、“style evaluation” 这一组关键词，保持主线统一。

不要把重点写成“我们实现了一个 LoRA 工程框架”。更好的表达是：我们围绕 style 信息完成了一个闭环，即从参考图中提取 style、把 style 压缩为动态 LightLoRA、将 style 注入扩散模型，并用可视化和指标验证 style 是否被表达出来。

如果篇幅有限，建议优先保留第 2、4、5、6、9、10 章，因为这些章节最直接体现“针对 style 信息的工作”。

## 依据材料

本大纲依据仓库中的 `README.md`、`README_evaluate.md`、核心模型代码、`image_in_ppt/` 展示结果、`outputs/` 评估结果和 `script.pdf` 的三页说明整理。当前仓库中未发现 `.tex` Beamer 源文件，因此没有直接引用 Beamer 源码结构。
