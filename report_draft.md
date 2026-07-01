# Learning Style Information as Dynamic LightLoRA Parameters

## Abstract

本文围绕单参考图风格迁移任务，研究如何从一张风格参考图中提取 style information，并将其转化为可注入扩散模型的动态参数。传统 DreamBooth 或 LoRA 方法通常需要针对每个风格单独训练，虽然可以获得较强的个性化效果，但训练时间、计算资源和参数存储成本较高。我们的工作尝试将参考图看作图像生成模型的视觉上下文，通过冻结的 DINOv2 编码器提取 patch-level visual tokens，再使用 HyperNetwork 将这些视觉特征映射为 LightLoRA 参数，并动态注入冻结的 Stable Diffusion 1.5 UNet。实验结果表明，相比不使用参考风格参数的 SD1.5 baseline，Image2LoRA 能够更明显地迁移参考图中的色调、纹理、材质和绘画感。在三组可视化样例上，平均 Gram style loss 从 0.00260 降至 0.00177，平均 AdaIN style loss 从 37.57 降至 24.04，说明参考图中的 style information 确实通过动态 LightLoRA 影响了生成分布。

## 1. Introduction

本文关注的问题不是简单生成一张视觉质量较好的图像，而是如何让生成图像继承单张参考图中的 style information。给定一张参考图和一段文本提示词，模型需要生成一张内容符合文本、视觉风格接近参考图的图像。这里的 style 并不等同于参考图中的具体物体或场景，而是包含颜色分布、纹理模式、材质质感、光照倾向、笔触结构和整体艺术化程度等更抽象的视觉信息。

现有个性化生成方法通常依赖 DreamBooth、Textual Inversion 或 LoRA 等微调技术。这类方法可以把某个特定主体或风格写入模型参数，但往往需要为每个新风格单独训练一套参数。如果目标是快速适配大量风格，这种逐风格训练方式会带来明显的计算和存储负担。因此，我们希望训练一个统一的模型，使其能够从单张参考图中提取 style representation，并在一次前向传播中生成可用于控制扩散模型的动态 LightLoRA 参数。

本文的核心问题可以概括为：单张参考图中的 style information 能否被有效提取、压缩成轻量参数、注入扩散模型，并在最终生成图像中稳定表达出来？

## 2. What is Style Information?

在本任务中，style information 可以从三个层次理解。第一是低层视觉统计，例如色调、饱和度、对比度和局部纹理。第二是中层视觉模式，例如材质、边缘形态、光照结构和笔触风格。第三是更高层的审美倾向，例如画面氛围、艺术化程度、构图偏好和抽象表达方式。

与 style 相对应的是 content。文本提示词主要控制 content，例如 “a river in the mountain” 决定图像中应出现山和河流；参考图则提供 style，例如整体颜色是否偏暖、纹理是否接近油画、材质是否柔和、画面是否具有插画感。理想的风格迁移模型应该在保留文本内容的同时，让输出图像的外观分布向参考图靠近。

因此，我们的工作重点不是复制参考图，而是学习参考图中可迁移的 style representation，并将这种表示转化为扩散模型内部的参数调制。

## 3. Motivation and Related Work

DreamBooth 和 LoRA 证明了扩散模型可以通过参数微调获得特定主体或风格的生成能力。LoRA 通过低秩矩阵更新降低了微调成本，但如果每个风格都需要单独训练 LoRA，整体流程仍然不够灵活。HyperDreamBooth 进一步说明，图像信息可以由 HyperNetwork 转换为模型参数更新，这为“从参考图直接预测风格参数”提供了启发。Lightweight DreamBooth 则提示我们，不一定需要直接预测完整 LoRA 矩阵，可以构造更小的参数子空间，从而降低 HyperNetwork 的输出维度。

受这些工作启发，我们将问题设定为 style-to-parameter mapping：输入不是风格标签，而是一张参考图；输出不是图像特征本身，而是可以注入扩散模型的动态 LightLoRA 参数。这样，style control 发生在模型参数层面，而不是仅依赖文本 prompt 或额外图像条件拼接。

## 4. Method

Image2LoRA 的整体流程由四个部分组成：style representation、HyperNetwork、LightLoRA 参数恢复和扩散模型注入。

首先，参考图被输入冻结的 DINOv2 编码器，得到 patch-level visual tokens。相比只使用全局图像 embedding，patch token 能保留更细粒度的局部视觉模式，例如纹理、色块、材质变化和笔触结构。然后，这些 visual tokens 被输入 transformer decoder 结构的 HyperNetwork。HyperNetwork 通过交叉注意力读取参考图特征，并经过多次 iterative refinement，生成每个目标注入层对应的 LightLoRA embedding。

接着，LightLoRA 模块不直接使用 HyperNetwork 输出完整 LoRA A/B 矩阵，而是将 embedding 与训练得到的 `down_aux` 和 `up_aux` 辅助矩阵组合，恢复实际的 LoRA 权重。最后，这些动态 LoRA 权重被注入冻结的 Stable Diffusion 1.5 UNet attention linear layers。文本提示词仍然控制图像语义内容，而参考图生成的动态 LoRA 参数负责调制图像风格。

方法流程可以概括为：

```text
Reference Image
    -> DINOv2 Patch Tokens
    -> HyperNetwork
    -> LightLoRA Embeddings
    -> Dynamic UNet Attention Modulation
    -> Stylized Image
```

## 5. LightLoRA as a Compact Style Carrier

LightLoRA 是本文承载 style information 的核心参数形式。普通 LoRA 需要为每个注入层生成 down 和 up 两个低秩矩阵。如果让 HyperNetwork 直接预测完整矩阵，输出维度较大，会增加训练难度，也容易降低泛化能力。本项目采用辅助矩阵设计：HyperNetwork 只预测一个较小的 embedding，再由该 embedding 和训练得到的辅助矩阵共同构造最终 LoRA 权重。

当前实现中，LightLoRA 的 rank 设置为 1，`down_dim=64`，`up_dim=32`。因此每层只需要预测 96 个标量。这相当于为 style information 设置了一个轻量 bottleneck：模型必须把参考图中对风格有用的信息压缩到小规模参数中，再通过 LoRA 的结构化注入影响 UNet。这样做的好处是参数量小、注入稳定，并且适合从单张参考图快速生成风格调制参数。

## 6. Training Pipeline

训练样本由参考风格图、目标图像和 caption 组成。训练时，参考图先经过 DINOv2 编码器得到 style representation，HyperNetwork 根据该表示生成动态 LightLoRA 参数。目标图像通过 VAE 编码到 latent space 后加噪，注入动态 LoRA 的冻结 SD1.5 UNet 需要在文本条件下预测噪声。训练目标是标准扩散噪声预测 MSE。

虽然损失函数仍然是扩散模型常用的噪声预测目标，但它对 HyperNetwork 施加了间接约束：HyperNetwork 必须学会从参考图中提取有助于恢复目标风格图分布的信息。训练过程中，Stable Diffusion 1.5 主体、VAE、文本编码器和 DINOv2 图像编码器基本冻结，主要训练 HyperNetwork 与 LightLoRA 辅助矩阵。因此，训练过程可以被理解为学习一个从 reference style information 到 diffusion parameter modulation 的映射。

## 7. Implementation

代码实现中，`DINOv2Encoder` 负责提取参考图 patch-level visual tokens。`ImageHyperDream` 和其中的 `ImageWeightGenerator` 负责将这些 style tokens 解码成每个注入层的 LightLoRA embedding。`LoRAModule` 使用 `down_aux`、`up_aux` 和 HyperNetwork 输出的 embedding 组合得到实际 LoRA 权重。`LoRANetwork` 将 LightLoRA 注入 SD1.5 UNet 的 attention linear layers，主要包括 `to_q`、`to_k`、`to_v` 和 `to_out.0`。

脚本层面，`scripts/train.py` 实现训练闭环，`scripts/infer.py` 支持输入单张参考图和文本提示词完成推理，`scripts/batch_infer.py` 用于批量生成，`scripts/evaluate.py` 用于计算 style loss、FID 等指标。

## 8. Experiments

实验设计围绕一个问题展开：生成结果是否真正利用了参考图中的 style information，而不是只根据文本提示词生成普通图像。

我们设置了两类实验。第一类是三组可视化案例，对比参考图、baseline SD1.5 输出和 Image2LoRA 输出。baseline 不使用参考图生成的 LoRA 参数，因此可以反映纯文本提示下的生成结果。第二类是批量评估实验，覆盖 9 个风格、3744 个生成样本，使用 FID 和 VGG style loss 衡量生成图像与参考风格之间的距离。

### 8.1 Visual Comparisons

**Case 1: a european style luxury palace**

| Reference | Baseline | Image2LoRA |
|---|---|---|
| ![palace reference](<image_in_ppt/a european style luxury palace/s0347____1008_01_query_1_img_000044_1683500663153_06774406385213314.jpeg.jpg>) | ![palace baseline](<image_in_ppt/a european style luxury palace/baseline.png>) | ![palace output](<image_in_ppt/a european style luxury palace/output.png>) |

在宫殿样例中，baseline 更像是根据文本生成的常规宫殿图像；Image2LoRA 的输出在整体色调、材质质感和艺术化程度上更接近参考图，说明参考图中的 style information 对生成分布产生了影响。

**Case 2: a mountain landscape in the style of the reference**

| Reference | Baseline | Image2LoRA |
|---|---|---|
| ![mountain reference](<image_in_ppt/a mountain landscape in the style of the reference/s1116____1101_01_query_1_img_000013_1684072792368_027720904633869425.jpeg.jpg>) | ![mountain baseline](<image_in_ppt/a mountain landscape in the style of the reference/baseline.png>) | ![mountain output](<image_in_ppt/a mountain landscape in the style of the reference/output.png>) |

在山景样例中，Image2LoRA 更明显地继承了参考图的绘画感和颜色倾向，而 baseline 的风格更接近通用扩散模型输出。这说明动态 LightLoRA 能够把参考图中的视觉上下文转化为有效的风格控制。

**Case 3: a river in the mountain**

| Reference | Baseline | Image2LoRA |
|---|---|---|
| ![river reference](<image_in_ppt/a river in the mountain/s0815____1012_01_query_0_img_000158_1683099524051_06904433845238462.jpg.jpg>) | ![river baseline](<image_in_ppt/a river in the mountain/baseline.png>) | ![river output](<image_in_ppt/a river in the mountain/output.png>) |

在河流样例中，两种方法都能生成符合文本的山水内容。Image2LoRA 的结果在整体视觉氛围和纹理表达上更受参考图影响，但该样例中 Gram style loss 的单项指标不完全优于 baseline，说明 style 迁移质量仍需要结合多指标和主观视觉观察共同判断。

### 8.2 Quantitative Metrics

三组可视化案例的 VGG style loss 汇总如下。Gram style loss 和 AdaIN style loss 都是越低越好。

| Method | Num Images | Gram Style Loss Mean | AdaIN Style Loss Mean |
|---|---:|---:|---:|
| Baseline | 3 | 0.00260 | 37.57 |
| Image2LoRA | 3 | 0.00177 | 24.04 |

逐案例指标如下：

| Prompt | Method | Gram Style Loss | AdaIN Style Loss |
|---|---|---:|---:|
| a european style luxury palace | Baseline | 0.00417 | 42.52 |
| a european style luxury palace | Image2LoRA | 0.00301 | 31.37 |
| a mountain landscape in the style of the reference | Baseline | 0.00333 | 57.02 |
| a mountain landscape in the style of the reference | Image2LoRA | 0.00194 | 30.46 |
| a river in the mountain | Baseline | 0.00030 | 13.15 |
| a river in the mountain | Image2LoRA | 0.00037 | 10.30 |

可以看到，Image2LoRA 在三组样例的平均 style loss 上均优于 baseline。其中 AdaIN style loss 从 37.57 降至 24.04，说明生成图像在 VGG 特征通道均值和方差统计上更接近参考图。Gram style loss 从 0.00260 降至 0.00177，说明特征相关性统计也整体更接近参考风格。

批量评估结果如下：

| Style ID | Num Cases | FID | Gram Style Loss | AdaIN Style Loss |
|---|---:|---:|---:|---:|
| s0055 | 420 | 355.11 | 0.00046 | 15.29 |
| s0112 | 420 | 341.48 | 0.00055 | 20.85 |
| s0129 | 420 | 319.78 | 0.00061 | 19.07 |
| s0134 | 420 | 406.43 | 0.00083 | 24.06 |
| s0162 | 420 | 351.42 | 0.00770 | 47.08 |
| s0172 | 420 | 324.19 | 0.00455 | 39.01 |
| s0188 | 420 | 307.72 | 0.00083 | 18.22 |
| s0205 | 420 | 328.15 | 0.00058 | 17.33 |
| s0234 | 384 | 341.93 | 0.00062 | 15.28 |
| **Average** | **3744** | **341.80** | **0.00186** | **24.02** |

批量结果说明，不同风格之间的迁移难度差异较大。例如 s0162 和 s0172 的 style loss 明显高于其他风格，可能说明这些风格包含更复杂或更难压缩的视觉模式。整体而言，平均 FID 为 341.80，平均 Gram style loss 为 0.00186，平均 AdaIN style loss 为 24.02，为后续更系统的风格评估提供了基准。

## 9. Discussion

实验结果表明，参考图中的 style information 可以通过动态 LightLoRA 改变生成分布。相比 baseline，Image2LoRA 更容易表现出参考图的色调、材质和绘画感。但 style transfer 本身并不是单一指标可以完全衡量的任务。一方面，模型需要迁移参考图风格；另一方面，它不能过度复制参考图而丢失文本内容。因此，style 和 content 之间存在天然 trade-off。

统一 HyperNetwork 的优势是快速、轻量、通用。给定一张新参考图，模型可以直接生成对应的 LightLoRA 参数，不需要重新训练一套 LoRA。但这种方式也可能在某些单一风格上不如专门微调的 LoRA 精细。换句话说，Image2LoRA 更适合被定位为一种 style parameter generation framework，而不是每个风格上的最优个性化微调方案。

## 10. Limitations

当前实验仍有若干限制。首先，现有评估缺少真实 content image，因此 SSIM、LPIPS 和 ArtFID 等内容一致性指标无法完整使用。其次，FID 对样本规模和参考集构造敏感，尤其在单风格参考数量有限时，不能单独代表风格迁移质量。第三，style information 本身包含多个层次，仅用 Gram loss 或 AdaIN loss 难以完整评价颜色、纹理、材质和审美倾向。第四，当前结果仍可能出现风格迁移不足、文本细节偏移或图像质量波动。

## 11. Future Work

后续工作可以继续围绕 style information 展开。第一，可以在 HyperNetwork 生成 LightLoRA 后加入短时间 rank-relaxed finetuning，以提升单张参考图上的风格细节。第二，可以比较不同 style representation 的效果，例如 DINO、CLIP、VGG 或多尺度图像特征。第三，可以系统研究 LoRA rank、注入层范围、辅助矩阵维度和 HyperNetwork 深度对 style 迁移质量的影响。第四，可以补充更完整的评估体系，包括 CLIP/DINO 特征相似度、CLIPScore、VQA-based score、KID、aesthetic score 和人类偏好实验。

## 12. Conclusion

本文草拟了一种以 style information 为核心的 Image2LoRA 报告主线。我们的工作把单张参考图看作视觉上下文，通过 DINOv2 提取 style representation，通过 HyperNetwork 完成 style-to-parameter mapping，并通过 LightLoRA 将 style information 注入冻结的 Stable Diffusion 1.5。实验结果显示，相比不使用参考图参数的 baseline，Image2LoRA 在可视化效果和平均 style loss 上都更接近参考图，说明参考图中的 style information 可以通过轻量参数调制影响扩散模型生成分布。

总体来看，Image2LoRA 提供了一条介于“纯 prompt 控制”和“逐风格训练 LoRA”之间的路线：它不需要为每个风格单独优化参数，却能够根据单张参考图动态生成风格调制权重。这为后续更大规模、更系统的单参考图风格生成研究提供了基础。
