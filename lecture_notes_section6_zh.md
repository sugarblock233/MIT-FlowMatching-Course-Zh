# 第 6 节 构建大规模图像与视频生成器（中文翻译）

对应原讲义 `lecture_notes.pdf` 第 41-53 页，即 Section 6：`Building Large-Scale Image or Video Generators`，包含 6.1、6.2、6.3 三个小节。

## 6 构建大规模图像与视频生成器

在前面的章节中，我们已经学会了如何训练一个 flow matching 或 diffusion model，从而从条件分布

$$
p_{\text{data}}(x\mid y)
$$

中采样。

这套方法本身是通用的，可以应用到很多不同的数据类型和任务中。这一节，我们会更深入地讨论大规模图像和视频生成这一类特别重要的应用场景，包括一些知名模型，例如 FLUX 2.0、Stable Diffusion 3、Nano Banana、VEO-3，以及 Meta Movie Gen Video。

最后，我们还会把前面学到的内容真正动手用起来，在实验课里从零搭建一个自己的版本。

这一节大致分成三部分：

1. 神经网络架构：先讨论如何把原始条件输入，包括时间 $t$ 和引导变量 $y_{\text{raw}}$（例如类别标签或原始文本），转换成模型 $u_t^\theta(x\mid y)$ 能够处理的向量表示；然后介绍常见架构，包括 U-Net 和 diffusion transformer。
2. Latent Space：介绍变分自编码器（VAE），它可以让我们在低维 latent space 中做生成建模，从而支持超高分辨率图像生成。
3. Case Study：最后分析两个代表性的大模型，Stable Diffusion 与 Meta Movie Gen，看看这些技术在真实大规模系统中是如何组合使用的。

## 6.1 神经网络架构

先来看：如果我们的目标是图像或视频这类“图像型模态”，那么应该怎样设计一个可扩展的神经网络架构来实现 flow / diffusion model？

更具体地说，我们要思考的是：实践中如何实现（guided）vector field

$$
u_t^\theta(x\mid y)
$$

这个函数。

注意，这个神经网络有 3 个输入：

- 一个向量 $x\in\mathbb{R}^d$
- 一个条件变量 $y\in\mathcal{Y}$
- 一个时间值 $t\in[0,1]$

并且有 1 个输出：

$$
u_t^\theta(x\mid y)\in\mathbb{R}^d.
$$

对于低维分布（例如前面章节里出现过的 toy distribution），直接把 $x,y,t$ 拼接起来，再喂进一个多层感知机（MLP，全连接神经网络）通常就够用了。

但对图像、视频、蛋白质这类复杂高维分布来说，MLP 往往不够强，因此通常要使用更适合具体任务的专门架构。

这一小节后面主要讨论图像（以及可扩展到视频）的情况。分成两步：

1. 先看怎样把原始条件信息，也就是时间 $t$ 和条件变量 $y$，变成可供模型处理的向量表示；
2. 再看两类常见架构：
   - U-Net
   - diffusion transformer（DiT）

### 6.1.1 条件变量的嵌入

#### 时间嵌入

对于简单 toy model，直接把标量时间 $t$ 拼到输入里就足够了。

但在实际中，人们通常会把这个标量时间先嵌入到更高维空间中，常见方法是 Fourier features。这样做的好处是：模型更容易表达高频率的时间依赖关系 [46]。

其具体形式是：

$$
\mathrm{TimeEmb}(t)
=
\sqrt{\frac{2}{d}}
\Bigl[
\cos(2\pi w_1 t),\ldots,\cos(2\pi w_{d/2} t),
\sin(2\pi w_1 t),\ldots,\sin(2\pi w_{d/2} t)
\Bigr]^\top.
\tag{68}
$$

其中频率 $w_i$ 取为

$$
w_i
=
w_{\min}
\left(\frac{w_{\max}}{w_{\min}}\right)^{\frac{i-1}{d/2-1}},
\qquad i=1,\ldots,d/2.
\tag{69}
$$

这种时间嵌入是一个标准选择，但并不是唯一必须的形式。它的一个优点是能方便地得到一个归一化的 $d$ 维表示，也就是

$$
\|\mathrm{TimeEmb}(t)\|=1,
$$

因为本质上利用了

$$
\sin^2+\cos^2=1.
$$

#### 类别标签嵌入

如果 $y_{\text{raw}}\in\mathcal{Y}\equiv\{0,\ldots,N\}$ 只是一个离散类别标签，那么最简单的方法就是：

> 为每个可能的标签值单独学习一个 embedding vector。

然后把这个向量作为 $y$ 输入模型。

这些 embedding 的参数也被算作 $u_t^\theta(x\mid y)$ 的参数的一部分，因此它们会和模型一起训练。

#### 文本输入嵌入

当 $y_{\text{raw}}$ 是文本提示词时，情况就更复杂了。现实中这通常依赖于冻结的、预训练好的模型。

这类模型的目标是把离散文本输入嵌入到一个连续向量空间里，并尽量保留文本中的关键信息。

一个经典例子是 CLIP（Contrastive Language-Image Pre-training）。CLIP 被训练成同时为图像和文本学习一个共享嵌入空间，它的损失函数会鼓励：

- 图像与其对应文本的 embedding 彼此接近；
- 与其他不匹配的图像或文本 embedding 相互远离 [34]。

因此，我们可以直接取

$$
y=\mathrm{CLIP}(y_{\text{raw}})\in\mathbb{R}^{d_{\text{CLIP}}}
$$

作为一个冻结的预训练 CLIP 模型输出的 embedding。

有时，把整段文本压缩成一个单向量并不理想。这种时候，还可以使用预训练 transformer，对整段提示词输出一个 embedding 序列。

而且在实际中，人们经常会把多种预训练 embedding 组合起来共同作为条件输入，以同时利用不同模型各自的优点 [14, 33]。

对我们来说，可以简单认为：经过这样的文本编码之后，prompt embedding 的形状是

$$
\mathrm{PromptEmbed}(y_{\text{raw}})\in\mathbb{R}^{S\times k}.
$$

### 6.1.2 Diffusion Transformers

在具体讲架构之前，先回忆一下：一张图像其实可以表示成

$$
x\in\mathbb{R}^{C_{\text{image}}\times H\times W},
$$

其中：

- $C_{\text{image}}$ 是通道数；
- 如果是 RGB 图像，通常 $C_{\text{image}}=3$；
- $H,W$ 分别是图像高度和宽度。

一类非常重要的架构叫做 diffusion transformer（DiT）及其变体 [12, 30, 28]。它们使用 attention 机制来构造模型 [49, 30, 28]。

DiT 有很多变种，这里讲义只介绍一个通用设计思路。不同具体模型在细节上可能会不同。

在这一小节里，记：

- $d$：hidden dimension
- $L$：transformer 层数
- $h$：每层的 attention head 数

DiT 的思想来自 vision transformer（ViT）：它会先把图像切成 patch，把 patch 映射成 token 序列，然后用标准 attention 处理这些 token [13]。最后再通过 depatchification 把输出还原回图像形状。

#### Patchify 与 Patch Embedding

最开始的 patchify 操作，其实就是把图像张量

$$
x\in\mathbb{R}^{C\times H\times W}
$$

重新组织成

$$
\mathrm{Patchify}(x)\in\mathbb{R}^{N\times C'},
$$

其中

$$
C'=CP^2,\qquad
N=(H/P)\cdot (W/P),
$$

$P$ 是 patch size。

然后再做一个线性映射，得到最终 patch embedding：

$$
\mathrm{PatchEmb}(x)=\mathrm{Patchify}(x)\,W\in\mathbb{R}^{N\times d},
$$

其中

$$
W\in\mathbb{R}^{C'\times d}
$$

是可学习参数。

#### DiT 的输入

于是 diffusion transformer 的三个输入变成：

$$
\tilde t = \mathrm{TimeEmb}(t)\in\mathbb{R}^d
$$

$$
\tilde y = \mathrm{PromptEmbed}(y)\in\mathbb{R}^{S\times d}
$$

$$
\tilde x_0 = \mathrm{PatchEmb}(x)\in\mathbb{R}^{N\times d}
$$

这样它们都已经具有 transformer 所需的 hidden dimension。

接着，DiT 通过一系列 transformer block 迭代更新 patch token：

$$
\tilde x_{i+1}
=
\mathrm{DiTBlock}(\tilde x_i,\tilde t,\tilde y)\in\mathbb{R}^{N\times d},
\qquad i=0,\ldots,L-1.
\tag{70}
$$

最后，再通过 depatchify 把输出变回图像形状：

$$
\tilde u=\mathrm{Depatchify}(\tilde x_N\tilde W)\in\mathbb{R}^{C\times H\times W},
$$

其中

$$
\tilde W\in\mathbb{R}^{d\times C'}.
$$

这个输出张量 $u$ 就是模型输出，也就是预测的速度场

$$
u_t^\theta(x\mid y).
$$

#### 注释 29（DiT Block）

讲义还简要给出了一层 DiT 的数学结构。这里保留核心思路。

设：

- 当前 patch token 序列为

$$
x\in\mathbb{R}^{N\times d}
$$

- 条件序列为

$$
y\in\mathbb{R}^{S\times d}
$$

那么一层典型的 DiT block 会做三件事：

1. patch token 上的 self-attention
2. 对 prompt 的 cross-attention
3. 通过 AdaLN（adaptive normalization）进行时间条件化

##### Scaled Dot Product Attention

给定 queries、keys、values：

$$
Q\in\mathbb{R}^{N\times d_h},\quad
K\in\mathbb{R}^{M\times d_h},\quad
V\in\mathbb{R}^{M\times d_h},
$$

则

$$
\mathrm{Attn}(Q,K,V)
=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
\in\mathbb{R}^{N\times d_h},
$$

其中 softmax 按行做。

##### Multi-Head Attention

设 head 数为 $h$，则每个 head 的维度是

$$
d_h=\frac{d}{h}.
$$

对每个 head，都学习投影矩阵

$$
W_Q^{(h)},W_K^{(h)},W_V^{(h)}\in\mathbb{R}^{k\times d_h}.
$$

定义

$$
\mathrm{head}_h(x,z)
=
\mathrm{Attn}(xW_Q^{(h)},zW_K^{(h)},zW_V^{(h)}).
$$

这里：

- 当 $z=x$ 时，就是 patch token 的 self-attention；
- 当 $z=y$ 时，就是对 prompt 的 cross-attention。

把所有 head 拼接后，再乘一个输出投影矩阵 $W_O\in\mathbb{R}^{d\times d}$，得到

$$
\mathrm{MultiHeadAttention}(x,z)
=
\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W_O
\in\mathbb{R}^{N\times d}.
$$

##### 时间条件化：Adaptive Normalization

令

$$
\tilde t\in\mathbb{R}^d
$$

为 timestep embedding。

DiT 中的标准做法，是利用 $\tilde t$ 生成逐通道的 scale / shift 参数，并用它们来调制归一化后的激活 [31]。

设

$$
g:\mathbb{R}^d\to \mathbb{R}^{2d}
$$

是一个 MLP，并令

$$
(\gamma,\beta)=g(\tilde t),
$$

其中 $\gamma,\beta\in\mathbb{R}^d$。

那么对 token 矩阵 $x\in\mathbb{R}^{N\times d}$ 和某个归一化算子 $\mathrm{Norm}(\cdot)$（例如 LayerNorm），定义

$$
\mathrm{AdaNorm}_{\tilde t}(x)
=
(1+\gamma)\odot \mathrm{Norm}(x)+\beta.
$$

其中 $\odot$ 表示按元素乘法，并对 token 维度做 broadcast。

##### 合起来

于是一个典型 DiT block 的更新大致是：

$$
x\leftarrow x+g_{\text{self}}(\tilde t)\odot
\mathrm{MultiHeadAttention}(\mathrm{AdaNorm}_{\tilde t}(x),\mathrm{AdaNorm}_{\tilde t}(x))
$$

$$
x\leftarrow x+g_{\text{cross}}(\tilde t)\odot
\mathrm{MultiHeadAttention}(\mathrm{AdaNorm}_{\tilde t}(x),y)
$$

$$
x\leftarrow x+g_{\text{MLP}}(\tilde t)\odot
\mathrm{MLP}(\mathrm{AdaNorm}_{\tilde t}(x))
$$

最终输出的 $x\in\mathbb{R}^{N\times d}$ 就作为下一层输入。

讲义还特别提醒：像实验里那种基于类别标签的 class-conditioned DiT，通常会更简单，不一定有 cross-attention，而更多是通过 time / class 的 AdaNorm 来做条件化。

### 6.1.3 U-Net

U-Net [38] 是另一类和 DiT 对应的重要架构，它本质上是一种特殊的卷积神经网络。

它最关键的特点在于：

> 输入和输出都具有图像的形状（只是通道数可能不同）。

这使它天然适合用来参数化向量场

$$
x\mapsto u_t^\theta(x\mid y),
$$

因为在固定 $y,t$ 时，输入是图像形状，输出也是图像形状。

因此，U-Net 在 diffusion model 早期文献中被极其广泛地使用 [17, 22, 11]。

一个 U-Net 由以下部分组成：

- 一系列 encoder $E_i$
- 对应的一系列 decoder $D_i$
- 以及位于中间的 latent processing block，讲义里把它称为 midcoder

举个例子，假设输入图像是

$$
x_t^{\text{input}}\in\mathbb{R}^{3\times 256\times 256}.
$$

在 U-Net 中，它会经历：

$$
x_t^{\text{input}}\in\mathbb{R}^{3\times 256\times 256}
\qquad\triangleright\ \text{输入}
$$

$$
x_t^{\text{latent}}=E(x_t^{\text{input}})
\in\mathbb{R}^{512\times 32\times 32}
\qquad\triangleright\ \text{经 encoder 压成 latent}
$$

$$
x_t^{\text{latent}}=M(x_t^{\text{latent}})
\in\mathbb{R}^{512\times 32\times 32}
\qquad\triangleright\ \text{通过 midcoder}
$$

$$
x_t^{\text{output}}=D(x_t^{\text{latent}})
\in\mathbb{R}^{3\times 256\times 256}
\qquad\triangleright\ \text{经 decoder 恢复输出}
$$

可以看到：

- encoder 过程中，通道数增加；
- 空间分辨率（高和宽）减小。

encoder 和 decoder 一般都由卷积层、激活函数、池化层等组成。

另外还有两个常见细节：

1. 输入图像在进入第一个 encoder 之前，通常会先经过一个预处理 block，把通道数先提上去；
2. encoder 与 decoder 之间往往有残差连接（skip / residual connection）。

U-Net 的名字也正是来自它整体呈现出的“U”形结构。

## 6.2 在 Latent Space 中工作：（变分）自编码器

到目前为止，我们一直在数据空间 $\mathbb{R}^d$ 中做建模。

但问题是：当图像分辨率越来越高时，直接在原始数据空间中建模的代价会迅速变得无法承受。

例如，一张 $1024\times 1024$、有 3 个 RGB 通道的图像，其总维度为

$$
d=H\cdot W\cdot 3 \approx 3\times 10^6.
$$

如果是视频，还要再乘上帧数 $T$，维度会更可怕。

与图像分类不同，分类任务的输出维度很低，而在 flow / diffusion 建模中，我们的输出

$$
u_t^\theta(x)\in\mathbb{R}^d
$$

必须和输入一样大。

因此，一个非常关键的问题是：

> 怎样在合理的显存和计算预算下，对超高维图像进行建模？

### 6.2.1 标准自编码器

一个自然的思路是压缩。

也许图像真正所在的“有效空间”，其实只是高维图像空间中的一个低维流形附近。于是我们可以考虑：

- 一个 encoder

$$
\mu_\phi:\mathbb{R}^d\to \mathbb{R}^k
$$

- 一个 decoder

$$
\mu_\theta:\mathbb{R}^k\to \mathbb{R}^d
$$

其中 $k\ll d$。

encoder 把原始图像 $x\in\mathbb{R}^d$ 编成 latent

$$
z\in\mathbb{R}^k,
$$

decoder 再把 latent 解回去。

例如，对图像

$$
d=3\times 1024\times 1024
$$

时，把它下采样成

$$
k=3\times \frac{1024}{16}\times \frac{1024}{16}
$$

这样的 latent，是很常见的做法。

这对 $(\mu_\phi,\mu_\theta)$ 合起来就叫 autoencoder。

理想情况下，我们希望 reconstruction 质量很好，也就是

$$
\mu_\theta(\mu_\phi(x))
$$

尽可能像原始 $x$。

因此，自编码器通常用 reconstruction loss 训练：

$$
L_{\text{Recon}}(\phi,\theta)
=
\mathbb{E}_{x\sim p_{\text{data}}}
\|\mu_\theta(\mu_\phi(x))-x\|^2.
$$

#### 是否适合生成建模

但只有 reconstruction loss 还不够。

因为我们的最终目标，是在 latent space 里训练一个生成模型，去学习 latent 分布

$$
p_{\text{latent}}(z),
\qquad
z=\mu_\phi(x),\ x\sim p_{\text{data}}.
$$

然后再把 latent 生成模型的输出送进 decoder $\mu_\theta$，从而得到对原始数据分布 $p_{\text{data}}(x)$ 的生成模型。

问题在于：用标准 autoencoder，我们几乎无法控制 $p_{\text{latent}}(z)$ 长什么样。

也就是说，虽然数据在 latent 空间里被压缩了，但 latent 分布本身未必“好学”。它可能仍然非常复杂、很不规则，甚至比原始数据分布还难学。

所以关键问题变成：

> 怎样保证 latent 分布 $p_{\text{latent}}$ 本身仍然是“规整的”、容易学习的？

为了更明确地正则化 latent 分布，我们接下来引入一个更一般的概率框架，也就是变分自编码器（variational autoencoder, VAE）。

### 6.2.2 变分自编码器

VAE 可以看作是对“标准（确定性）自编码器”的一个放松：encoder 和 decoder 不再是确定函数，而是概率分布。

具体来说，考虑：

- 编码器

$$
q_\phi(z\mid x)
$$

- 解码器

$$
p_\theta(x\mid z)
$$

最常见的形式是高斯分布：

$$
q_\phi(z\mid x)=\mathcal{N}(z;\mu_\phi(x),\mathrm{diag}(\sigma_\phi^2(x))),
$$

$$
p_\theta(x\mid z)=\mathcal{N}(x;\mu_\theta(z),\sigma_\theta^2(z)I_d).
\tag{71}
$$

其中：

- $\mu_\phi(x)\in\mathbb{R}^k$
- $\sigma_\phi^2(x)\in\mathbb{R}_{\ge 0}^k$
- $\mu_\theta(z)\in\mathbb{R}^d$
- $\sigma_\theta^2(z)\in\mathbb{R}_{\ge 0}$

都由神经网络参数化。

于是：

- 编码时，采样

$$
z\sim q_\phi(\cdot\mid x)
$$

- 解码时，采样

$$
x\sim p_\theta(\cdot\mid z)
$$

如果总有

$$
\sigma_\phi(x)=0,\qquad \sigma_\theta(z)=0,
$$

那么就退化回标准 autoencoder。

#### VAE 的 reconstruction loss

在这种随机编码 / 解码的设定下，一个自然的 reconstruction 目标是：

$$
L_{\text{VAE-Recon}}(\phi,\theta)
=
-\mathbb{E}_{x\sim p_{\text{data}}(x),\,z\sim q_\phi(\cdot\mid x)}
\bigl[\log p_\theta(x\mid z)\bigr].
\tag{72}
$$

和标准自编码器相比，有两点变化：

1. 编码结果不再是确定的，而是从 $q_\phi(z\mid x)$ 采样；
2. reconstruction 误差不再是简单 MSE，而是把原始样本 $x$ 在解码分布下的负对数似然作为损失。

对高斯情形，这个 reconstruction loss 可写成：

$$
L_{\text{VAE-Recon}}(\phi,\theta)
=
\mathbb{E}_{x\sim p_{\text{data}}(x),\,z\sim q_\phi(z\mid x)}
\left[
\frac{1}{2\sigma_\theta^2(z)}\|x-\mu_\theta(z)\|^2
+
\frac{d}{2}\log \sigma_\theta^2(z)
+
\text{const}
\right].
\tag{73}
$$

因此，VAE 的 reconstruction loss 本质上和标准 autoencoder 还是很接近的，只不过需要把所有可能的编码结果 $z$ 都考虑进去。

其中第二项和 decoder 的方差有关，它反映了 reconstruction 精度与预测不确定性之间的权衡。

在很多实现里，包括实验里，$\sigma_\phi(x)$ 和 $\sigma_\theta(z)$ 都会被简化成可学习的标量常数，而不是依赖于 $x,z$ 的完整函数。这样能避免数值不稳定和一些退化行为。

在这种简化下，VAE reconstruction loss 基本就退化成“带随机编码的 MSE”：

$$
L_{\text{VAE-Recon}}(\phi,\theta)
=
\mathbb{E}_{x\sim p_{\text{data}}(x),\,z\sim q_\phi(z\mid x)}
\left[
\frac{1}{2\sigma_\theta^2}\|x-\mu_\theta(z)\|^2+\text{const}
\right].
\tag{74}
$$

#### 给 latent 分布加先验约束

现在重新回到我们的目标：

> 我们希望数据经过编码映射到 latent space 之后，latent 分布是一个“规整的、好学的”分布。

为此，我们引入一个 latent prior 分布

$$
p_{\text{prior}}(z).
$$

在这里，通常取

$$
p_{\text{prior}}=\mathcal{N}(0,I_k),
$$

也就是各向同性高斯。

这个 prior 代表了我们理想中希望 latent 分布接近的目标：一个简单、规整、容易学习的分布。

于是，我们对 encoder 加一个辅助正则项：

$$
L_{\text{VAE-Prior}}(\phi)
=
\mathbb{E}_{x\sim p_{\text{data}}(x)}
\Bigl[
D_{\mathrm{KL}}(q_\phi(\cdot\mid x)\|p_{\text{prior}})
\Bigr].
\tag{75}
$$

这里的 $D_{\mathrm{KL}}$ 是 Kullback-Leibler divergence（KL 散度），它是衡量两个概率分布差异的标准工具。

这个损失的含义很直观：

> 我们希望对任意数据点 $x$，其编码分布 $q_\phi(z\mid x)$ 都尽量像一个高斯先验。

如果这一点对所有 $x$ 都成立，那么自然就有理由期待整体 latent 分布也接近高斯。

#### 注释 30（KL 散度背景）

对两个概率密度 $q,p$，KL 散度定义为

$$
D_{\mathrm{KL}}(q(x)\|p(x))
=
\int q(x)\log \frac{q(x)}{p(x)}\,dx
=
\mathbb{E}_{X\sim q}
\left[
\log \frac{q(X)}{p(X)}
\right].
$$

它有两个重要性质：

$$
D_{\mathrm{KL}}(q\|p)\ge 0
\tag{76}
$$

以及

$$
D_{\mathrm{KL}}(q\|p)=0
\Longleftrightarrow
q=p.
\tag{77}
$$

也就是说：

- KL 散度永远非负；
- 只有当两个分布完全相同时，它才为 0。

#### VAE 总损失

于是，把 reconstruction loss 与 prior loss 按一个权重 $\beta\ge 0$ 结合起来，就得到 VAE 的训练目标：

$$ 
L_{\text{VAE}}(\phi,\theta)
=
L_{\text{VAE-Recon}}(\phi,\theta)
+
\beta L_{\text{VAE-Prior}}(\phi)
\tag{78}
$$

也就是

$$
L_{\text{VAE}}(\phi,\theta)
=
-\mathbb{E}_{x\sim p_{\text{data}}(x),\,z\sim q_\phi(z\mid x)}
\bigl[\log p_\theta(x\mid z)\bigr]
+
\beta\,
\mathbb{E}_{x\sim p_{\text{data}}(x)}
\bigl[
D_{\mathrm{KL}}(q_\phi(\cdot\mid x)\|p_{\text{prior}})
\bigr].
\tag{79}
$$

其中：

- 第一项保证 latent 能够被 decoder 有效还原回数据；
- 第二项保证 latent 分布接近高斯 prior。

$\beta$ 控制两者之间的权衡。

#### 例 31（各向同性高斯之间的 KL 散度）

设

$$
q(x)=\mathcal{N}(x;\mu_q,\mathrm{diag}(\sigma_q^2)),
\qquad
p(x)=\mathcal{N}(x;\mu_p,\mathrm{diag}(\sigma_p^2)).
$$

则有

$$
D_{\mathrm{KL}}(q\|p)
=
\frac{1}{2}K\left(\frac{\sigma_q^2}{\sigma_p^2}\right)
+
\frac{1}{2}\frac{\|\mu_q-\mu_p\|^2}{\sigma_p^2},
\tag{80}
$$

其中

$$
K(\alpha)=\sum_{i=1}^d (\alpha_i-\log \alpha_i-1).
$$

这个表达式很直观：

- 当均值和方差都完全一致时，$D_{\mathrm{KL}}(q\|p)=0$；
- 均值相差越大，KL 越大；
- $K(\alpha)$ 在 $\alpha=1$ 时取得最小值，因此当 $\sigma_q=\sigma_p$ 时最优。

在讲义原文的推导里，还会先经过一个中间式：

$$
D_{\mathrm{KL}}(q\|p)
=
\frac{1}{2}\log\frac{\sigma_p^2}{\sigma_q^2}
+
\frac{1}{2\sigma_p^2}\,\mathbb{E}_q\|x-\mu_p\|^2
-
\frac{1}{2\sigma_q^2}\,\mathbb{E}_q\|x-\mu_q\|^2.
\tag{81}
$$

#### 高斯 encoder 下的 prior loss

如果 encoder 是高斯形式，那么

$$
L_{\text{VAE-Prior}}(\phi)
=
\mathbb{E}_{x\sim p_{\text{data}}(x)}
\bigl[
D_{\mathrm{KL}}(q_\phi(\cdot\mid x)\|\mathcal{N}(0,I_k))
\bigr]
$$

会化成

$$
=
\mathbb{E}\left[
\frac{1}{2}K(\sigma_\phi^2(x))
+
\frac{1}{2}\|\mu_\phi(x)\|^2
\right].
\tag{82}
$$

这个损失也很好理解：

- 它惩罚 latent mean 偏离 0；
- 惩罚 latent variance 偏离 1。

于是整体 VAE 损失就可以写成

$$
L_{\text{VAE}}(\phi,\theta)
=
\mathbb{E}_{x\sim p_{\text{data}}(x),\,z\sim q_\phi(z\mid x)}
\Bigl[
\frac{1}{2\sigma_\theta^2(z)}\|x-\mu_\theta(z)\|^2
+
\frac{d}{2}\log \sigma_\theta^2(z)
+
\frac{\beta}{2}K(\sigma_\phi^2(x))
+
\frac{\beta}{2}\|\mu_\phi(x)\|^2
\Bigr].
\tag{83}
$$

这四项分别对应：

1. reconstruction error
2. decoder uncertainty
3. 让 latent variance 接近 1
4. 让 latent mean 接近 0

### 训练 VAE

接下来还剩一个问题：

> 怎样最小化 VAE 损失？

麻烦在于：损失中的期望是对

$$
q_\phi(z\mid x)
$$

取的，而这个分布本身依赖于参数 $\phi$。

这时就需要经典的 reparameterization trick（重参数化技巧）。

对于

$$
q_\phi(z\mid x)=\mathcal{N}(z;\mu_\phi(x),\sigma_\phi^2(x)I_k),
$$

我们可以这样采样：

$$
\epsilon\sim \mathcal{N}(0,I_k),\qquad
z=\mu_\phi(x)+\sigma_\phi(x)\epsilon
\quad\Longrightarrow\quad
z\sim q_\phi(\cdot\mid x).
$$

这样一来，随机性只来自

$$
\epsilon\sim \mathcal{N}(0,I_k),
$$

而它的分布与 $\phi$ 无关。

因此，VAE 损失就能重写成标准深度学习框架可以直接优化的形式：

$$
L_{\text{VAE}}(\phi,\theta)
=
\mathbb{E}_{x\sim p_{\text{data}}(x),\,\epsilon\sim \mathcal{N}(0,I_k)}
\Bigl[
\frac{1}{2\sigma_\theta^2(z)}
\|x-\mu_\theta(\mu_\phi(x)+\sigma_\phi(x)\epsilon)\|^2
+
\frac{d}{2}\log \sigma_\theta^2(z)
+
\frac{\beta}{2}K(\sigma_\phi^2(x))
+
\frac{\beta}{2}\|\mu_\phi(x)\|^2
\Bigr].
$$

如果再进一步把 $\sigma_\theta^2(z)$ 固定成常数 $\sigma^2$，那么损失还可以再简化成：

$$
L_{\text{VAE}}(\phi,\theta)
=
\mathbb{E}_{x\sim p_{\text{data}}(x),\,\epsilon\sim \mathcal{N}(0,I_k)}
\Bigl[
\frac{1}{2\sigma^2}
\|x-\mu_\theta(\mu_\phi(x)+\sigma_\phi(x)\epsilon)\|^2
+
\frac{\beta}{2}K(\sigma_\phi^2(x))
+
\frac{\beta}{2}\|\mu_\phi(x)\|^2
\Bigr].
$$

### 算法 6：$\beta$-VAE 训练流程

设 decoder 为高斯分布，且方差固定：

$$
p_\theta(x\mid z)=\mathcal{N}(x;\mu_\theta(z),\tilde\sigma^2 I_d).
$$

输入：

- 数据集 $x\sim p_{\text{data}}$
- encoder 网络 $(\mu_\phi(x),\log \sigma_\phi^2(x))$
- decoder 网络 $\mu_\theta(z)$
- latent 维度 $k$
- 常数 $\beta\ge 0,\ \tilde\sigma^2>0$

训练步骤：

1. 对每个 mini-batch $\{x_i\}_{i=1}^B$
2. 编码：

$$
\mu_i\leftarrow \mu_\phi(x_i),\qquad
\log \sigma_i^2\leftarrow \log \sigma_\phi^2(x_i)
$$

3. 采样噪声

$$
\epsilon_i\sim \mathcal{N}(0,I_k)
$$

4. 重参数化：

$$
z_i\leftarrow \mu_i+\sigma_i\odot \epsilon_i,
\qquad
\sigma_i=\exp\left(\frac{1}{2}\log \sigma_i^2\right)
$$

5. 解码均值：

$$
\hat x_i\leftarrow \mu_\theta(z_i)
$$

6. reconstruction loss：

$$
L_{\text{recon}}
=
\frac{1}{B}\sum_{i=1}^B
\frac{1}{2\tilde\sigma^2}\|x_i-\hat x_i\|^2
$$

7. 到 prior $p_{\text{prior}}(z)=\mathcal{N}(0,I_k)$ 的 KL loss：

$$
L_{\text{KL}}
=
\frac{1}{B}\sum_{i=1}^B
\frac{1}{2}
\sum_{j=1}^k
(\mu_{i,j}^2+\sigma_{i,j}^2-\log \sigma_{i,j}^2-1)
$$

8. 总损失：

$$
L=L_{\text{recon}}+\beta L_{\text{KL}}
$$

9. 用梯度更新 $(\phi,\theta)$

### 实践备注

上面的推导展示的是 autoencoder / VAE 的基本原则。实际中，人们常常还会加入额外损失或约束。讲义里给了几个很实用的工程提醒：

1. 选择 $\beta$（以及 KL warm-up）
   - 较大的 $\beta$ 会让 latent 更接近 prior，但可能伤害 reconstruction，甚至引发 posterior collapse；
   - 常见的稳定化做法是 KL warm-up：训练一开始先设 $\beta=0$，再逐步升高；
   - 不过现代很多 autoencoder 的 $\beta$ 实际上都非常小，也就是 $\beta\ll 1$。

2. decoder variance
   - 学习一个完整的高斯 decoder variance $\sigma_\theta^2$ 在数值上可能比较脆弱；
   - 因此很多实现会直接固定

$$
p_\theta(x\mid z)=\mathcal{N}(x;\mu_\theta(z),\sigma^2 I_d),
$$

这样 reconstruction term 基本就是 MSE（差个常数）。

3. 超越像素级 MSE 的 reconstruction loss
   - 对图像来说，单纯像素 MSE 往往会导致重建过于平滑；
   - 实践中常会加 perceptual loss（例如在预训练网络的特征空间上比较）。

4. 对抗式 / 混合目标
   - 为了进一步提升视觉真实性，还可以把 VAE 目标和 adversarial loss 结合起来（类似 VAE-GAN）；
   - 这样通常会让图像更锐利，但也会带来额外的不稳定性和更多超参数。

### 注释 32（在 Latent Space 中工作）

一旦有了 autoencoder，训练 latent generative model 的方法其实很直接：

- 训练时：
  从 $x\sim p_{\text{data}}$ 出发，经 encoder 采样得到 latent $z\sim q_\phi(z\mid x)$，然后直接在 latent space 上训练 flow / diffusion model。

- 推理时：
  先从 latent diffusion / flow model 中采样得到 $z$，再用 decoder 解码：

$$
x=\mu_{\text{mean}}(z),
$$

这里通常取 decoder 的均值，而不是再随机采样，以避免额外噪声带来伪影。

一个训练良好的 autoencoder，可以被理解为：

> 它把高频或语义上不重要的细节过滤掉了，让生成模型把注意力集中在更重要、更感知相关的特征上。

在讲义写作时，几乎所有最先进的图像和视频生成方法，都采用了所谓 latent diffusion 范式：先训练 autoencoder，再在它的 latent space 中训练 flow / diffusion model [36, 48]。

不过也要记住：

> 最终系统效果不只取决于 diffusion / flow model，也非常依赖 autoencoder 本身压缩和恢复图像的能力。

更多关于 VAE 的讨论，讲义放在了附录 D。

## 6.3 Case Study：Stable Diffusion 3 与 Meta Movie Gen

这一节最后用两个大型生成模型做简短 case study：

- Stable Diffusion 3：图像生成
- Meta Movie Gen Video：视频生成

它们都会用到本讲义介绍的技术，同时也加入了一些更复杂的工程增强，以支持更大规模训练和更复杂的条件输入（例如文本）。

### 6.3.1 Stable Diffusion 3

Stable Diffusion 是一系列最先进的图像生成模型，也是最早大规模使用 latent diffusion 做图像生成的模型之一。

如果你没试过，讲义很推荐你自己上网体验一下：

[https://stability.ai/news/stable-diffusion-3](https://stability.ai/news/stable-diffusion-3)

Stable Diffusion 3 使用的训练目标，就是我们这份讲义里讨论的 conditional flow matching（讲义中提到 Algorithm 4，不过从上下文看对应的是前面的训练配方）。

论文中他们测试了多种 flow / diffusion 变体，最后发现 flow matching 效果最好。

它的训练方式还包括：

- classifier-free guidance training
- 在 pre-trained autoencoder 的 latent space 中训练

事实上，早期 stable diffusion 论文的一个重要贡献，就是训练出了高质量 autoencoder，使 latent diffusion 真正可用。

为了提升文本条件化能力，Stable Diffusion 3 同时使用了 3 种不同的文本 embedding，包括：

- CLIP embedding
- Google T5-XXL encoder 的序列输出 [35]
- 以及与 [3, 39] 类似的一些组合做法

这些 embedding 的分工大致是：

- CLIP 提供一种整体、粗粒度的语义表示；
- T5 提供更细粒度的文本上下文，使模型能够关注提示词中的具体成分。

为了接收这种序列型文本上下文，作者进一步扩展了 diffusion transformer：它不仅对图像 patch 做 attention，也对文本 embedding 做 attention。

这种改造后的 DiT 被称为 multi-modal DiT（MM-DiT）。

它最大的模型有：

$$
8\ \text{billion parameters}.
$$

采样时，Stable Diffusion 3 使用：

- 50 步 Euler 模拟
- classifier-free guidance scale 约在 2.0 到 5.0 之间

也就是说，采样一张图时，网络大概要被评估 50 次。

### 6.3.2 Meta Movie Gen Video

接下来是 Meta 的视频生成器 Movie Gen Video：

[https://ai.meta.com/research/movie-gen/](https://ai.meta.com/research/movie-gen/)

这里的数据不再是图像，而是视频，因此数据 $x$ 位于空间

$$
\mathbb{R}^{T\times C\times H\times W},
$$

其中 $T$ 是时间维度，也就是视频帧数。

从这个例子里你会看到：很多在图像场景里的设计（例如 autoencoder、DiT）其实都被改造成了能够处理“多一个时间维度”的版本。

Movie Gen Video 使用的训练目标仍然是 conditional flow matching，并采用直线型 scheduler：

$$
\alpha_t=t,\qquad \sigma_t=1-t.
$$

和 Stable Diffusion 3 一样，它也在一个冻结的、预训练好的 autoencoder latent space 中工作。

而且对视频来说，autoencoder 的意义甚至更大，因为视频的内存开销远大于图像。因此目前绝大多数视频生成器在视频长度上都还比较受限。

具体而言，作者提出了 temporal autoencoder（TAE），把原始视频

$$
x_t'\in\mathbb{R}^{T'\times 3\times H\times W}
$$

映射成 latent

$$
x_t\in\mathbb{R}^{T\times C\times H\times W},
$$

其中讲义里给出的压缩比例是

$$
\frac{T'}{T}
=
\frac{H'}{H}
=
\frac{W'}{W}
=
8.
$$

为了处理长视频，作者还提出了 temporal tiling：把长视频切成多个片段，分别编码，再把 latent 拼接起来 [33]。

模型主体 $u_t^\theta(x_t)$ 使用的是类似 DiT 的骨干网络，只不过 patchify 不仅沿空间维度切，也沿时间维度切。

随后，这些 patch token 会进入一个 transformer，其中包括：

- patch 之间的 self-attention
- 对语言模型 embedding 的 cross-attention

这点和 Stable Diffusion 3 里的 MM-DiT 很像。

在文本条件化方面，Movie Gen Video 使用了 3 类文本 embedding：

- UL2 embedding：用于细粒度的文本推理 [47]
- ByT5 embedding：用于捕捉字符级细节，例如 prompt 里明确要求出现某些文字时 [50]
- MetaCLIP embedding：在共享图文嵌入空间中训练得到 [24, 33]

它最大的模型规模达到：

$$
30\ \text{billion parameters}.
$$

如果你想看更完整、更工程化的细节，讲义也建议直接阅读 Movie Gen 的技术报告 [33]。
