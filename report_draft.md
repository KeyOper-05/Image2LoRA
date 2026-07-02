# Learning Style Information as Dynamic LightLoRA Parameters

## Abstract

Generative image models can now synthesize diverse visual content from text, but controlling how an image looks remains as important as controlling what it depicts. Style transfer addresses this problem by asking a model to preserve the semantic content specified by a prompt while adopting visual properties, such as color, texture, material appearance, lighting, and painterly quality, from a reference image. Existing personalization methods, including DreamBooth and LoRA, can encode a specific style into model parameters, but they usually require separate optimization for each new style and therefore become costly when many styles must be handled. A central challenge is whether style information from a single reference image can be extracted once, compressed into lightweight model-side parameters, and used immediately at inference time. Here we show that a frozen DINOv2 encoder and a HyperNetwork can map patch-level reference-image features into dynamic LightLoRA parameters that modulate the UNet denoiser inside frozen Stable Diffusion 1.5. Compared with a text-only SD1.5 baseline, Image2LoRA produces outputs that more closely reflect the reference image in tone, texture, material appearance, and stylization. 

## 1. Introduction

Text-to-image diffusion models have become a powerful tool for visual synthesis, as they can generate diverse images from natural-language prompts while preserving broad semantic controllability \cite{rombach2022ldm}. However, text prompts are often insufficient for specifying how an image should look. In many practical creative tasks, users do not only want to describe the content of an image, but also want to control its visual style: the color palette, texture, material appearance, lighting tendency, brushstroke structure, and overall artistic quality. This makes style transfer and style-conditioned generation important problems in modern image generation.

Style transfer has a long history in neural image synthesis. Early neural style transfer methods showed that deep visual features can separate and recombine content and style, and that Gram-matrix statistics of convolutional features can represent important aspects of artistic style \cite{gatys2015neural}. Later arbitrary style transfer methods, such as AdaIN, further showed that matching feature-level statistics can provide a practical mechanism for transferring style between images \cite{huang2017adain}. In our setting, the model is given a reference image and a text prompt. The prompt specifies the content to generate, while the reference image provides style information. The goal is not to copy the objects or layout of the reference image, but to transfer abstract visual properties from the reference while preserving the semantic content requested by text.

Existing text-to-image personalization methods provide one way to encode such visual information into generative models. Textual Inversion learns new text embeddings for user-provided concepts \cite{gal2022textual}; DreamBooth fine-tunes a pretrained text-to-image model to bind a subject or style to a rare identifier \cite{ruiz2022dreambooth}; and LoRA reduces the cost of model adaptation by injecting trainable low-rank matrices into a frozen model \cite{hu2021lora}. These methods demonstrate that subject or style information can be stored in model-side parameters, but they still usually require a separate optimization process for each new concept or style. This becomes inefficient when the goal is to adapt quickly to many styles or to use a single reference image at inference time.

To reduce this per-style training cost, we treat style information as the visual context of a generation task. This viewpoint is inspired by the role of context in language models: context provides task-specific information, but it can also be compressed into model-side representations or parameter updates. In image generation, a style reference image can similarly be viewed as visual context that contains task-specific information about color, texture, material, and artistic appearance. HyperDreamBooth has shown that HyperNetworks can generate personalized diffusion-model weights from image inputs \cite{ruiz2023hyperdreambooth}, suggesting that reference images can be mapped directly into parameter updates. Following this direction, we use a frozen DINOv2 encoder to extract patch-level visual features from the reference image \cite{oquab2023dinov2}, and a HyperNetwork to compress those features into dynamic LightLoRA parameters.

In this project, we aim to train a unified model that extracts a style representation from a single reference image and generates LightLoRA parameters in one forward pass. During inference, these parameters are injected into the frozen UNet denoiser of Stable Diffusion 1.5, so that the text prompt controls image content while the reference-conditioned LightLoRA controls visual style. In this way, our work studies whether style information can be represented as compact dynamic parameters, providing a lightweight alternative between prompt-only control and per-style LoRA fine-tuning.

## 2. Motivation and Related Work

**Text-to-image diffusion models.** Recent text-to-image generators are commonly diffusion models, which learn to generate images by reversing a gradual noising process. Latent Diffusion Models made this process more efficient by performing denoising in a compressed latent space rather than directly in pixel space, while still supporting text conditioning through cross-attention in the denoising network \cite{rombach2022ldm}. Stable Diffusion 1.5 follows this latent diffusion paradigm and provides a practical backbone for controlled generation experiments. Here, we choose SD1.5 in this project because it is more vram-friendly than newer, larger diffusion backbones, making it more suitable for limited computational resources. This choice allows us to focus on the core question of style representation and dynamic parameter generation, rather than on scaling the base model itself.

**Neural style transfer.** The idea of separating image content from image style originates from neural style transfer. Gatys et al. showed that deep features can preserve high-level content, while Gram matrices of feature activations capture important style statistics \cite{gatys2015neural}. AdaIN later made arbitrary style transfer more efficient by aligning channel-wise feature statistics between content and style images \cite{huang2017adain}. These works motivate our evaluation choices: even though style is difficult to define with a single metric, VGG-based Gram statistics and AdaIN statistics provide useful proxies for measuring how close an output image is to a reference style.

**Text-to-image personalization and parameter-efficient adaptation.** Personalization methods attempt to make a pretrained generative model reproduce a specific concept, subject, or style. Textual Inversion represents a user-provided concept through learned token embeddings in the text space \cite{gal2022textual}. DreamBooth fine-tunes a text-to-image model so that a rare identifier becomes associated with a specific subject or appearance \cite{ruiz2022dreambooth}. LoRA provides a more parameter-efficient adaptation mechanism by freezing the base model and learning low-rank updates \cite{hu2021lora}. These methods demonstrate that visual information can be encoded into either text-side embeddings or model-side parameters. Nevertheless, they usually require a separate optimization process for each new concept or style. For single-reference style transfer, this is a major limitation: the user may want to provide an arbitrary reference image at inference time without waiting for per-style training.

**HyperNetworks** HyperNetworks provide a way to generate model parameters from input conditions. In the personalization setting, HyperDreamBooth showed that a HyperNetwork can generate a small set of personalized weights from a single face image, greatly reducing the cost of DreamBooth-style adaptation \cite{ruiz2023hyperdreambooth}. This suggests a more general strategy: instead of treating the reference image only as an external condition, we can map it into parameter updates that directly modulate the generative model. Our project follows this parameter-generation direction, but shifts the focus from identity personalization to style information. The output of our HyperNetwork is not a full fine-tuned model, but compact LightLoRA parameters that are injected into selected attention projection layers inside the SD1.5 UNet denoiser.

**Long-context compression** A useful analogy comes from long-context modeling in language models. Long-context methods study how to preserve task-relevant information when the raw context is too large to use directly. Compressive Transformer compresses past memories for long-range sequence modeling \cite{rae2019compressive}, while LLMLingua compresses long prompts to reduce inference cost while preserving essential information \cite{jiang2023llmlingua}. More directly related to our formulation, SHINE maps meaningful context into LoRA adapters for language models in a single forward pass \cite{liu2026shine}. These works motivate the view that context can be transformed into parameter-side knowledge. In our case, the reference image is the context, style is the information to preserve, and LightLoRA is the compressed parameter representation.

Inspired by these lines of work, we formulate our task as style-to-parameter mapping. The input is not a discrete style label or a set of optimization steps, but a single reference image. The output is not merely an image embedding used as conditioning, but a set of dynamic LightLoRA parameters that can be injected into a diffusion model. In this way, style control happens at the model-parameter level, providing a lightweight alternative to both prompt-only control and per-style LoRA fine-tuning.

## 3. Methodology

### 3.1 Overview

The central idea of Image2LoRA is to represent style information as dynamic model-side parameters. Instead of using the reference image only as an external condition, we convert it into LightLoRA weights that directly modulate certain layers of a frozen diffusion model. This design separates the roles of the two inputs: the text prompt specifies semantic content, while the reference image provides style information that is compressed into parameter space.

The full pipeline consists of five steps: (1)reference-image encoding, (2)HyperNetwork decoding, (3)LightLoRA weight recovery, (4)LoRA injection, and (5)denoising-based training or inference. The reference image is encoded once into patch-level visual tokens. A HyperNetwork reads these tokens and generates one compact LightLoRA embedding for each target injection layer. Each embedding is then expanded into a low-rank update through learned auxiliary matrices. During generation, the resulting dynamic LoRA weights are injected into selected attention projection layers of the SD1.5 UNet denoiser, so that the full SD1.5 pipeline generates images with text-controlled content and reference-controlled style.



### 3.2 Style Representation from Reference Images

We use a frozen DINOv2 encoder to encode patch-level visual tokens from the reference image. DINOv2 is designed to produce robust visual features without task-specific supervision \cite{oquab2023dinov2}. Although these features are often used for semantic representation, their patch tokens also preserve local visual information such as color regions, repeated textures, material transitions, and edge or brushstroke patterns. These properties are useful for style transfer because style is not only a global label, but a spatially distributed set of visual cues.

Let the reference image be denoted as \(I_{\mathrm{ref}}\). The frozen image encoder maps it into a token sequence:

\[
F_{\mathrm{ref}} = E_{\mathrm{DINO}}(I_{\mathrm{ref}}) \in \mathbb{R}^{L \times d},
\]

where \(L\) is the number of patch tokens and \(d\) is the feature dimension. We interpret \(F_{\mathrm{ref}}\) as the style representation used by the rest of the system. The encoder is frozen so that training focuses on learning how to translate visual style features into diffusion-model parameter modulation rather than adapting the feature extractor itself.

### 3.3 HyperNetwork for Style-to-Parameter Mapping

The HyperNetwork implements the style-to-parameter mapping. Its input is the reference-image token sequence \(F_{\mathrm{ref}}\), and its output is a set of compact LightLoRA embeddings. Each embedding corresponds to one LoRA injection layer in the UNet. 

The decoder uses transformer blocks with cross-attention over the DINOv2 tokens. It starts from zero-initialized layer-wise weight tokens and refines them iteratively. Positional embeddings distinguish different LoRA target layers, while cross-attention allows each layer token to read the reference image features. After several refinement steps, the HyperNetwork produces a matrix of LightLoRA embeddings:

\[
Z = H(F_{\mathrm{ref}}) \in \mathbb{R}^{N \times m},
\]

where \(N\) is the number of injected LoRA layers and \(m\) is the embedding dimension for each layer.

### 3.4 LightLoRA as a Compact Style Carrier

LightLoRA carries style information. A standard LoRA module applies a low-rank update to a frozen linear layer:

\[
W' = W + \Delta W, \qquad \Delta W = BA,
\]

where \(W\) is the original weight matrix and \(A,B\) are trainable low-rank matrices. Directly predicting all entries of \(A\) and \(B\) for every target layer would make the HyperNetwork output large and difficult to train. Instead, following the convention of HyperDreamBooth, Image2LoRA predicts a compact embedding \(z_i\) for each target layer \(i\), and combines it with learned auxiliary matrices to recover the actual LoRA update.

The key difference between LightLoRA and ordinary LoRA is the use of auxiliary matrices. For a target linear layer with input dimension \(d_{\mathrm{in}}\) and output dimension \(d_{\mathrm{out}}\), each `LoRAModule` stores two learned auxiliary matrices:

\[
A_{\mathrm{aux}} \in \mathbb{R}^{d_{\mathrm{down}} \times d_{\mathrm{in}}},
\qquad
B_{\mathrm{aux}} \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{up}}}.
\]

These matrices define a shared parameter subspace for possible LoRA updates. Instead of predicting full LoRA matrices of shape \(r \times d_{\mathrm{in}}\) and \(d_{\mathrm{out}} \times r\), the HyperNetwork predicts only two small coefficient matrices:

\[
C_{\mathrm{down}} \in \mathbb{R}^{r \times d_{\mathrm{down}}},
\qquad
C_{\mathrm{up}} \in \mathbb{R}^{d_{\mathrm{up}} \times r}.
\]

The effective LoRA projections are recovered by multiplying the predicted coefficients with the auxiliary matrices:

\[
A = C_{\mathrm{down}} A_{\mathrm{aux}},
\qquad
B = B_{\mathrm{aux}} C_{\mathrm{up}}.
\]

The final update is then:

\[
\Delta W = BA =
(B_{\mathrm{aux}} C_{\mathrm{up}})
(C_{\mathrm{down}} A_{\mathrm{aux}}).
\]

In this formulation, the auxiliary matrices are trainable, style-agnostic basis matrices, while the HyperNetwork output provides style-specific coordinates inside this basis. This is important for single-reference generation: the model does not need to predict a full high-dimensional update from one image. It only needs to predict where the reference style lies in a learned low-dimensional LoRA subspace.

### 3.5 Training Objective

We use the standard diffusion $\epsilon$-prediction objective. Each training sample contains a reference style image, a target image, and a caption. The reference image is encoded by DINOv2, the HyperNetwork generates LightLoRA parameters, and these parameters are injected into the frozen UNet. The target image is encoded into the latent space by the VAE encoder and noised at a randomly sampled timestep. The UNet then predicts the noise under the text condition and the generated LightLoRA modulation.

Let \(x_0\) be the VAE latent of the target image, \(t\) the sampled timestep, \(\epsilon\) the Gaussian noise, and \(x_t\) the noised latent. The training loss is:

\[
\mathcal{L} =
\mathbb{E}_{x_0,t,\epsilon}
\left[
\left\|
\epsilon -
\epsilon_\theta(x_t,t,c_{\mathrm{text}};\Delta W(I_{\mathrm{ref}}))
\right\|_2^2
\right],
\]

where \(c_{\mathrm{text}}\) is the text condition and \(\Delta W(I_{\mathrm{ref}})\) denotes the dynamic LightLoRA weights. During training, the SD1.5 VAE, text encoder, UNet denoiser, and DINOv2 encoder are frozen; the main trainable parts are the HyperNetwork and the LightLoRA auxiliary matrices.

## 4. Implementation

### 4.1 Model Components

The implementation follows the methodology above with four main modules. `DINOv2Encoder` extracts patch-level visual tokens from the reference image. `ImageHyperDream` and its `ImageWeightGenerator` decode these tokens into one LightLoRA embedding per injected layer. `LoRAModule` converts each compact embedding into effective LoRA weights by combining it with `down_aux` and `up_aux`. `LoRANetwork` creates the target LoRA modules and injects them into the UNet denoiser of Stable Diffusion 1.5.

Inside `LoRAModule`, `down_aux` has shape `(down_dim, in_features)` and `up_aux` has shape `(out_features, up_dim)`. The generated `weight_embedding` is split into two parts: a down-side coefficient of shape `(rank, down_dim)` and an up-side coefficient of shape `(up_dim, rank)`. The code computes `down = down_weight @ down_aux` and `up = up_aux @ up_weight`, then applies the two projections sequentially to the model weights.

We choose SD1.5 as our text-to-image backbone, which consists mainly of a text encoder, a VAE, a scheduler, and a UNet denoiser. The target modules are the linear projection layers inside attention modules of the UNet `Transformer2DModel` blocks: `to_q`, `to_k`, `to_v`, and `to_out.0`.

### 4.2 Training

The training loop loads the tokenizer, text encoder, VAE, UNet, and scheduler from the Stable Diffusion 1.5 checkpoint. The VAE, text encoder, UNet, and DINOv2 encoder are frozen. A LightLoRA network is created and applied to the UNet, and the HyperNetwork is initialized with the same number of output tokens as the number of injected LoRA layers.

For each batch, the script encodes the reference image to obtain DINOv2 features, generates layer-wise LightLoRA embeddings, updates the LoRA modules with these generated weights, encodes the target image into latent space, samples a diffusion timestep, adds noise, and trains the UNet prediction path with the denoising MSE objective. Checkpoints store two types of parameters: `hypernetwork.safetensors` for the HyperNetwork and `lora_aux.safetensors` for the LightLoRA auxiliary matrices.

### 4.3 Inference

Given a reference image and a prompt, the script first loads the frozen Stable Diffusion 1.5 components and the trained HyperNetwork/LightLoRA auxiliary weights. It then encodes the reference image with DINOv2, generates dynamic LightLoRA embeddings, updates the LoRA modules in the UNet, and runs the Stable Diffusion pipeline with the user prompt.

During inference, no optimization is performed for the new reference image. Style adaptation happens through a single forward pass of the image encoder and HyperNetwork. This is what distinguishes Image2LoRA from per-style LoRA training.

## 5. Experiments

### Evaluation

We conduct two types of experiments. In the qualitative evaluation, we compare the reference image, the SD1.5 baseline output, and the Image2LoRA output on three visual examples. The baseline does not use reference-conditioned LoRA parameters, so it reflects generation under text-only conditioning. In the quantitative evaluation, we evaluate 9 styles and 3744 generated samples using VGG-based style losses and FID to measure style-statistical and distributional similarity to the reference set.

### 5.1 Visual Comparisons

**Case 1: a european style luxury palace**

\begin{figure}[t]
  \centering
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{image_in_ppt/a european style luxury palace/s0347____1008_01_query_1_img_000044_1683500663153_06774406385213314.jpeg.jpg}
    \caption{Reference}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{image_in_ppt/a european style luxury palace/baseline.png}
    \caption{Baseline}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{image_in_ppt/a european style luxury palace/output.png}
    \caption{Image2LoRA}
  \end{subfigure}
  \caption{Visual comparison for the prompt ``a european style luxury palace''.}
  \label{fig:palace}
\end{figure}

In the palace example, the baseline looks more like a conventional palace generated from the text prompt. The Image2LoRA output is closer to the reference image in global tone, material appearance, and degree of stylization, indicating that reference style information influences the generated distribution.

**Case 2: a mountain landscape in the style of the reference**

\begin{figure}[t]
  \centering
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{image_in_ppt/a mountain landscape in the style of the reference/s1116____1101_01_query_1_img_000013_1684072792368_027720904633869425.jpeg.jpg}
    \caption{Reference}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{image_in_ppt/a mountain landscape in the style of the reference/baseline.png}
    \caption{Baseline}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{image_in_ppt/a mountain landscape in the style of the reference/output.png}
    \caption{Image2LoRA}
  \end{subfigure}
  \caption{Visual comparison for the prompt ``a mountain landscape in the style of the reference''.}
  \label{fig:mountain}
\end{figure}

In the mountain landscape example, Image2LoRA better inherits the painterly feeling and color tendency of the reference image, while the baseline is closer to a generic diffusion model output. This suggests that dynamic LightLoRA can convert visual context from the reference image into effective style control.

**Case 3: a river in the mountain**

\begin{figure}[t]
  \centering
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{image_in_ppt/a river in the mountain/s0815____1012_01_query_0_img_000158_1683099524051_06904433845238462.jpg.jpg}
    \caption{Reference}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{image_in_ppt/a river in the mountain/baseline.png}
    \caption{Baseline}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{image_in_ppt/a river in the mountain/output.png}
    \caption{Image2LoRA}
  \end{subfigure}
  \caption{Visual comparison for the prompt ``a river in the mountain''.}
  \label{fig:river}
\end{figure}

In the river example, both methods generate content consistent with the prompt. Image2LoRA shows stronger influence from the reference image in visual atmosphere and texture, but its Gram style loss is not lower than the baseline for this single case. This indicates that style transfer quality should be judged using multiple metrics together with visual inspection.

### 5.2 Quantitative Metrics

The VGG-based style loss results for the three visual examples are summarized in Table~\ref{tab:style-loss-summary}. Both Gram style loss and AdaIN style loss are lower-is-better metrics, following the common use of VGG feature statistics in neural style transfer \cite{simonyan2014vgg,gatys2015neural,huang2017adain}.

Unlike many style transfer models that transfer an existing content image into a reference style, our method generates a stylized image from a reference image and a text prompt. Thus, the current experimental data contains reference style images, baseline outputs, and Image2LoRA outputs, but it does not provide strictly paired content images for each case. Therefore, many common content-fidelity or holistic style-transfer metrics cannot be directly applied. For example, SSIM requires a content image to measure structural similarity \cite{wang2004ssim}, LPIPS is typically used to measure perceptual distance between generated and content images \cite{zhang2018lpips}, and ArtFID was proposed as a neural style transfer metric that combines content fidelity with style distribution quality \cite{wright2022artfid}. Since content images are unavailable in the current setup, these metrics are skipped.

It is also worth noting that style loss does not have a single standard implementation in the same way that FID has a widely used evaluation package and protocol \cite{heusel2017fid}. Thus, the absolute values may not be directly comparable with those reported in other studies. In this repository, we implement two VGG-based style distances following common neural style transfer and AdaIN definitions. Specifically, we use Gram style loss, which compares Gram matrices of multi-layer VGG features, and AdaIN style loss, which compares channel-wise means and standard deviations of VGG features \cite{gatys2015neural,huang2017adain}.

\begin{table}[t]
  \centering
  \caption{Average VGG-based style losses on the three visual examples. Lower values indicate closer style statistics to the reference images.}
  \label{tab:style-loss-summary}
  \begin{tabular}{lccc}
    \toprule
    Method & Num Images & Gram Style Loss & AdaIN Style Loss \\
    \midrule
    Baseline & 3 & 0.00260 & 37.57 \\
    Image2LoRA & 3 & 0.00177 & 24.04 \\
    \bottomrule
  \end{tabular}
\end{table}

Per-case results are listed in Table~\ref{tab:style-loss-cases}.

\begin{table}[t]
  \centering
  \caption{Per-case VGG-based style losses for the three visual examples.}
  \label{tab:style-loss-cases}
  \begin{tabular}{lcccc}
    \toprule
    & \multicolumn{2}{c}{Gram Style Loss} & \multicolumn{2}{c}{AdaIN Style Loss}\\
       \cmidrule(r){2-3} \cmidrule(r){4-5}
    Prompt & Baseline & Image2LoRA & Baseline & Image2LoRA \\
    \midrule
    palace & 0.00417 & 0.00301 & 42.52 & 31.37 \\
    mountain landscape & 0.00333 & 0.00194 & 57.02 & 30.46 \\
    river in the mountain & 0.00030 & 0.00037 & 13.15 & 10.30 \\
    \bottomrule
  \end{tabular}
\end{table}



On average, Image2LoRA outperforms the baseline in style loss across the three examples. The AdaIN style loss decreases from 37.57 to 24.04, suggesting that the generated images are closer to the reference images in terms of VGG feature channel statistics. The Gram style loss decreases from 0.00260 to 0.00177, indicating that feature correlation statistics are also closer to the reference style overall.

The batch evaluation results are shown in Table~\ref{tab:batch-eval}.

\begin{table}[t]
  \centering
  \caption{Batch evaluation over 9 styles and 3744 generated samples. Lower FID and style losses indicate closer distributional or style-statistical similarity to the reference set.}
  \label{tab:batch-eval}
  \begin{tabular}{lrrrr}
    \toprule
    Style ID & Num Cases & FID & Gram Style Loss & AdaIN Style Loss \\
    \midrule
    s0055 & 420 & 355.11 & 0.00046 & 15.29 \\
    s0112 & 420 & 341.48 & 0.00055 & 20.85 \\
    s0129 & 420 & 319.78 & 0.00061 & 19.07 \\
    s0134 & 420 & 406.43 & 0.00083 & 24.06 \\
    s0162 & 420 & 351.42 & 0.00770 & 47.08 \\
    s0172 & 420 & 324.19 & 0.00455 & 39.01 \\
    s0188 & 420 & 307.72 & 0.00083 & 18.22 \\
    s0205 & 420 & 328.15 & 0.00058 & 17.33 \\
    s0234 & 384 & 341.93 & 0.00062 & 15.28 \\
    \midrule
    Average & 3744 & 341.80 & 0.00186 & 24.02 \\
    \bottomrule
  \end{tabular}
\end{table}

The batch results show that style transfer difficulty varies across styles. For example, s0162 and s0172 have noticeably higher style loss than most other styles, which may indicate that these styles contain more complex or less easily compressed visual patterns. Overall, the average FID is 341.80, the average Gram style loss is 0.00186, and the average AdaIN style loss is 24.02. These values provide an initial reference point for future systematic style evaluation. We provide these reference style images in Appendix for better understanding.

The current FID computation compares the distribution of generated images with the distribution of reference style images \cite{heusel2017fid}. However, FID mainly measures distributional distance. When the evaluation set is small, FID can be high because the estimated feature statistics are strongly affected by content and layout differences rather than by abstract style alone. Thus, under our experimental setting, a high FID does not necessarily indicate failed style transfer. Instead, it may be partly attributed to differences in content and layout, because our model generates images from text prompts rather than transforming paired reference content images.

Several factors should be considered when interpreting these evaluation results. First, Image2LoRA performs style adaptation in a single forward pass without per-reference fine-tuning, so it may be less optimized for a specific style than a separately trained LoRA. At the same time, the model must generate content that remains faithful to the text prompt while incorporating style information from the reference image. This makes the task different from many traditional style transfer settings, where the model is given a content image and does not need to synthesize the layout from scratch. Therefore, it is reasonable for the generation quality to be lower than that of traditional style transfer models of similar size. Second, due to limited computational resources, we use SD1.5 as the base diffusion backbone. Its UNet is relatively small compared with newer diffusion architectures, so some remaining imperfections in style transfer may be alleviated by training the HyperNetwork with a stronger text-to-image backbone. Third, because no paired content images are available, the generated image distribution may be affected not only by style but also by differences in layout, object composition, and prompt semantics, leading to high FID between reference and generated samples. Finally, as the absolute values of style loss are implementation-dependent,they should only be compared within the same evaluation protocol.

## 6. Discussion

The experiments suggest that reference style information can change the generated distribution through dynamic LightLoRA. Compared with the baseline, Image2LoRA more clearly reflects the tone, material appearance, texture, and painterly quality of the reference image. However, style transfer cannot be fully evaluated by a single metric. The model must transfer the reference style while still preserving prompt-specified content. If style control is too strong, semantic details may be weakened; if it is too weak, the output becomes close to the text-only baseline. Therefore, there is an inherent trade-off between style and content.

The unified HyperNetwork is fast, lightweight, and general. Given a new reference image, it can directly generate corresponding LightLoRA parameters without retraining a new LoRA. However, for some individual styles, it may be less precise than a LoRA specifically fine-tuned for that style. Thus, Image2LoRA is better positioned as a style parameter generation framework rather than an optimal per-style personalization method.

## 7. Limitations

The current experiments have several limitations. First, our setting does not provide paired content images as input, so content-fidelity metrics such as SSIM, LPIPS, and ArtFID cannot be applied in their standard form. As a result, the quantitative evaluation mainly focuses on style distribution and VGG-based style statistics. Second, FID is sensitive to sample size and the construction of the reference set, especially when only a limited number of reference images are available for each style. In our setting, the absolute FID value should therefore be interpreted cautiously, because it may reflect differences in content and layout rather than style transfer quality alone. Third, style loss does not have a universally standardized implementation. The Gram style loss and AdaIN style loss used in this report are project-level metrics based on common VGG feature statistics, making them useful for relative comparison within our experiments but less suitable for direct comparison across different studies. Finally, the current model is still an initial prototype and may suffer from incomplete style transfer, loss of fine text-driven details, or unstable image quality in some cases.

## 8. Future Work

Future work can further investigate how style information should be represented, compressed, and injected into diffusion models. First, a short rank-relaxed fine-tuning stage could be applied after HyperNetwork-generated LightLoRA to recover finer style details for a specific reference image. Second, different visual representations could be compared more systematically, including DINO, CLIP, VGG, and multi-scale image features, to better understand which features are most suitable for style extraction. Third, key design choices in LightLoRA and the HyperNetwork should be studied in more detail, including LoRA rank, injection layer range, auxiliary matrix dimension, and decoder depth. Finally, scaling the method to stronger diffusion backbones and larger training sets would help clarify both the potential and the limitations of HyperNetwork-based style parameter generation.

## 9. Conclusion

This report presents Image2LoRA from the perspective of style information. The project treats a single reference image as visual context, extracts a style representation with DINOv2, maps this representation into dynamic parameters through a HyperNetwork, and injects the resulting LightLoRA into attention projection layers inside the frozen SD1.5 UNet denoiser. Experimental results show that, compared with a baseline without reference-conditioned parameters, Image2LoRA produces outputs that are visually and statistically closer to the reference style.

Overall, Image2LoRA provides a middle route between pure prompt-based control and per-style LoRA training. It does not require optimizing a separate parameter set for every style, yet it can generate style modulation weights dynamically from a single reference image. This provides a foundation for future work on scalable and systematic single-reference style generation.

\appendix
\section{Batch Evaluation Reference Images}

\begin{figure}[t]
  \centering
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{sampled_50styles_500pairs_package/style/s0055____1019_01_query_0_img_000004_1684018013820_07938481893127348.jpg.jpg}
    \caption{s0055}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{sampled_50styles_500pairs_package/style/s0112____1025_01_query_1_img_000156_1683664012862_022263192689277955.jpeg.jpg}
    \caption{s0112}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{sampled_50styles_500pairs_package/style/s0129____1110_01_query_2_img_000025_1683693126351_07183911352804041.jpeg.jpg}
    \caption{s0129}
  \end{subfigure}

  \vspace{0.8em}

  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{sampled_50styles_500pairs_package/style/s0134____1110_01_query_0_img_000083_1683080483399_013692774561724075.jpeg.jpg}
    \caption{s0134}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{sampled_50styles_500pairs_package/style/s0162____0919_01_query_0_img_000045_1683863628251_0564223797649242.jpeg.jpg}
    \caption{s0162}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{sampled_50styles_500pairs_package/style/s0172____1107_01_query_1_img_000018_1683099425957_08968873238001791.jpg.jpg}
    \caption{s0172}
  \end{subfigure}

  \vspace{0.8em}

  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{sampled_50styles_500pairs_package/style/s0188____0919_01_query_1_img_000027_1682858602199_07286026440159936.jpeg.jpg}
    \caption{s0188}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{sampled_50styles_500pairs_package/style/s0205____1101_01_query_2_img_000020_1683343384788_014901948185305225.jpeg.jpg}
    \caption{s0205}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{sampled_50styles_500pairs_package/style/s0234____0925_01_query_1_img_000086_1682692809469_0651319034038563.jpg.jpg}
    \caption{s0234}
  \end{subfigure}

  \caption{Reference style images used in the batch evaluation. The nine images correspond to the nine style IDs reported in Table~\ref{tab:batch-eval}.}
  \label{fig:batch-eval-style-refs}
\end{figure}
