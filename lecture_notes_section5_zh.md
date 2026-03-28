# 第 5 节 Guidance：如何按提示词进行条件生成（中文翻译）

对应原讲义 `lecture_notes.pdf` 第 34-40 页，即 Section 5：`Guidance: How To Condition on a Prompt`，包含 5.1、5.2 两个小节。

## 5 Guidance：如何按提示词进行条件生成

到目前为止，我们讨论的生成模型都是“无引导”的。比如，一个图像模型只是随便生成一张图。

从数学上说，这意味着模型是在从一个无条件数据分布

$$
p_{\text{data}}(z)
$$

中采样。

但在大多数实际场景中，我们的目标并不是生成任意对象，而是生成一个满足某些附加信息条件的对象。换句话说，我们希望“引导”模型去生成某一类特定对象。

例如，你可以设想一个图像生成模型，它接受一个文本提示词 $y$，然后生成一张与这个提示词匹配的图像 $x$。正如第 1 节中讨论过的，这意味着我们希望从条件数据分布中采样：

$$
p_{\text{data}}(z\mid y),
$$

也就是在 $y$ 的条件下的数据分布。

这一节就来讨论这个问题。

### 注释 25（术语说明）

为了避免和前面“condition on $z\sim p_{\text{data}}$”的用法混淆（例如 conditional probability path / conditional vector field），这里我们专门用 guided 这个词来表示“对 $y$ 做条件化”，其中 $y$ 例如可以是一个文本提示词。

## 5.1 Vanilla Guidance

我们先来看“标准版”的条件生成方式。

简短的答案是：

> 在训练和推理时，直接把提示词 $y$ 作为网络输入的一部分，其余一切保持和之前一样。

下面把它形式化。

我们把条件变量或提示词 $y$ 视为属于某个空间 $\mathcal{Y}$。

- 如果 $y$ 是文本提示词，那么 $\mathcal{Y}$ 就是所有文本组成的空间；
- 如果 $y$ 是离散类别标签，那么 $\mathcal{Y}$ 就是一个离散集合。

这里我们不对 $\mathcal{Y}$ 做任何额外限制。

我们定义一个 guided diffusion model，由以下两部分组成：

1. 一个 guided vector field $u_t^\theta(\cdot\mid y)$，由神经网络参数化；
2. 一个随时间变化的扩散系数 $\sigma_t$。

形式上写为

$$
u^\theta:\mathbb{R}^d\times \mathcal{Y}\times [0,1]\to \mathbb{R}^d,
\qquad
(x,y,t)\mapsto u_t^\theta(x\mid y),
$$

以及固定的

$$
\sigma_t:[0,1]\to [0,\infty),\qquad t\mapsto \sigma_t.
$$

注意它和前面 Summary 7 的区别：这里神经网络除了输入 $x,t$ 之外，还额外接收了引导变量 $y\in\mathcal{Y}$。

给定任意 $y\in\mathcal{Y}$，我们就可以按下面的方式生成样本：

- 初始化：

$$
X_0\sim p_{\text{init}}
$$

- 模拟：

$$
dX_t = u_t^\theta(X_t\mid y)\,dt+\sigma_t\,dW_t
$$

- 目标：

$$
X_1\sim p_{\text{data}}(\cdot\mid y)
$$

也就是说：

- 从一个简单分布（比如高斯分布）开始初始化；
- 把 SDE 从 $t=0$ 模拟到 $t=1$；
- 希望最终样本 $X_1$ 服从条件分布 $p_{\text{data}}(\cdot\mid y)$。

当 $\sigma_t=0$ 时，这样的模型就叫 guided flow model。

接下来，为了叙述简洁，我们主要以 flow matching 和 flow model 为例，但同样的思想也适用于更一般的 diffusion model。

现在的问题是：

> 我们应该怎样训练一个 guided flow model $u_t^\theta(x\mid y)$？

一个直接的思路是：固定某个给定的 $y$，然后把数据分布看成

$$
p_{\text{data}}(x\mid y).
$$

这样一来，我们其实就把条件生成重新变成了一个无条件生成问题，因此仍然可以像之前一样，用 conditional flow matching 目标来训练：

$$
\mathbb{E}_{z\sim p_{\text{data}}(\cdot\mid y),\,x\sim p_t(\cdot\mid z)}
\bigl[
\|u_t^\theta(x\mid y)-u_t^{\text{target}}(x\mid z)\|^2
\bigr].
\tag{57}
$$

注意这里的标签 $y$ 并不会影响条件概率路径 $p_t(\cdot\mid z)$，也不会影响条件向量场 $u_t^{\text{target}}(x\mid z)$（当然原则上也可以让它们依赖于 $y$，但这里没有这么做）。

再对所有可能的 $y$ 展开期望，就得到 guided conditional flow matching 的训练目标：

$$
L_{\mathrm{CFM}}^{\mathrm{guided}}(\theta)
=
\mathbb{E}_{(z,y)\sim p_{\text{data}}(z,y),\ t\sim \mathrm{Unif}[0,1],\ x\sim p_t(\cdot\mid z)}
\bigl[
\|u_t^\theta(x\mid y)-u_t^{\text{target}}(x\mid z)\|^2
\bigr].
\tag{58}
$$

这个 guided 目标和第 3 节里无引导目标（式 (26)）的一个主要区别在于：

- 无引导时，我们只采样 $z\sim p_{\text{data}}$
- 这里则要采样联合样本 $(z,y)\sim p_{\text{data}}(z,y)$

原因很简单：此时我们的数据分布本质上是图像 $z$ 和文本提示词 $y$ 的联合分布。

在实现上，这意味着比如在 PyTorch 里，dataloader 返回的不再只是 $z$，而是同时返回 $z$ 和 $y$。

## 5.2 Classifier-Free Guidance

从理论上看，vanilla guidance 本应足以生成服从

$$
p_{\text{data}}(\cdot\mid y)
$$

的样本。

但很快人们在实验中发现：仅用这种方式生成的图像，对提示词 $y$ 的贴合程度还不够好。

这可能有很多原因：

1. 模型欠拟合，也就是没有真正学到目标边缘向量场；
2. 数据本身不完美，比如互联网上的图文配对数据经常有大量噪声和错误。

因此，如果我们真的想让样本更符合提示词，就需要一种办法去“人为放大”提示词 $y$ 的作用。

最主要的技术，就是 classifier-free guidance（CFG）。它被广泛用于当前最先进的 diffusion model 中。下面我们来讨论它。

### Classifier Guidance

为了说明问题，我们先只考虑高斯概率路径的情形。

回忆一下，第 3 节里高斯条件概率路径写成

$$
p_t(\cdot\mid z)=\mathcal{N}(\alpha_t z,\beta_t^2 I_d),
$$

其中 $\alpha_t,\beta_t$ 是连续可微、单调的噪声调度函数，并满足

$$
\alpha_0=\beta_1=0,\qquad
\alpha_1=\beta_0=1.
$$

同时，由命题 1，我们可以把 guided vector field 写成 guided score 的形式：

$$
u_t^{\text{target}}(x\mid y)=a_t \nabla \log p_t(x\mid y)+b_t x.
\tag{59}
$$

接着注意到：$p_t(x\mid y)$ 是一个条件密度，因此可以使用 Bayes 公式：

$$
p_t(x\mid y)=\frac{p_t(x)p_t(y\mid x)}{p_t(y)}.
\tag{60}
$$

对它取对数再对 $x$ 求梯度，有

$$
\nabla \log p_t(x\mid y)
=
\nabla \log \frac{p_t(x)p_t(y\mid x)}{p_t(y)}
=
\nabla \log p_t(x)+\nabla \log p_t(y\mid x),
\tag{61}
$$

因为梯度是对变量 $x$ 求的，而 $p_t(y)$ 与 $x$ 无关，所以

$$
\nabla \log p_t(y)=0.
$$

因此，我们可以把 guided vector field 改写为

$$
u_t^{\text{target}}(x\mid y)
=
b_t x+a_t\bigl(\nabla \log p_t(x)+\nabla \log p_t(y\mid x)\bigr)
=
u_t^{\text{target}}(x)+a_t\nabla \log p_t(y\mid x).
$$

这个式子的结构很有意思：

> guided vector field = unguided vector field + 一个关于提示变量 $y$ 的分类器梯度项

由于人们观察到生成样本对提示词的服从度不够好，一个很自然的想法就是放大这个

$$
\nabla \log p_t(y\mid x)
$$

项的作用，于是得到

$$
\tilde u_t(x\mid y)
=
u_t^{\text{target}}(x)+w a_t\nabla \log p_t(y\mid x),
\qquad
\text{(classifier guidance)}
\tag{62}
$$

其中 $w>1$ 被称为 guidance scale（引导强度）。

那这个 $\log p_t(y\mid x)$ 怎么学呢？

注意，它其实可以被看成一个“对加噪数据做分类”的分类器：它给出了在给定 $x$ 时，标签 $y$ 的对数似然。因此，我们完全可以用监督学习去单独训练它。

这就得到了 classifier guidance [11, 43]。

不过后来，classifier guidance 基本被 classifier-free guidance 取代了，所以讲义不再深入展开它。它更像是 classifier-free guidance 的思想基础。

最后还要强调一点：这其实是一个启发式方法（heuristic）。当

$$
w\neq 1
$$

时，有

$$
\tilde u_t(x\mid y)\neq u_t^{\text{target}}(x\mid y),
$$

也就是说它已经不是“真正”的 guided vector field 了。

### Classifier-Free Guidance

虽然 classifier guidance 在理论上可行，但它存在几个现实问题：

1. 我们需要额外训练一个分类器，因此不再是 1 个网络，而是 2 个网络；
2. 如果 $y$ 是高维的，比如一整段文本，而不是一个简单类别，那么 $p_t(y\mid x)$ 很难学，$\nabla \log p_t(y\mid x)$ 也很难得到。

因此，人们提出了 classifier-free guidance（CFG）[18]。它在效果上与 classifier guidance 等价，但不需要单独训练一个 classifier。

做法如下。仍然利用

$$
\nabla \log p_t(x\mid y)
=
\nabla \log p_t(x)+\nabla \log p_t(y\mid x),
$$

于是

$$
\tilde u_t(x\mid y)
=
u_t^{\text{target}}(x)+w a_t\nabla \log p_t(y\mid x)
$$

$$
=
u_t^{\text{target}}(x)+w a_t\bigl(\nabla \log p_t(x\mid y)-\nabla \log p_t(x)\bigr)
$$

$$
=
(1-w)u_t^{\text{target}}(x)+w u_t^{\text{target}}(x\mid y).
$$

因此，放大后的 guided vector field 可以表示为：

> unguided vector field 与 guided vector field 的线性组合

也就是说，理论上我们只要同时得到

- 无条件向量场 $u_t^{\text{target}}(x)$
- 有条件向量场 $u_t^{\text{target}}(x\mid y)$

在推理时把它们按上式组合起来，就能得到强化过的

$$
\tilde u_t(x\mid y).
$$

这时你可能会问：

> 那不还是要训练两个模型吗？

好消息是，不需要。

我们可以把标签空间扩充一下，加入一个新的特殊标签

$$
\varnothing
$$

用来表示“没有条件”。然后把无条件向量场写成

$$
u_t^{\text{target}}(x)=u_t^{\text{target}}(x\mid \varnothing).
$$

这样一来，我们就可以用同一个模型同时学习：

- 条件模型 $u_t^{\text{target}}(x\mid y)$
- 无条件模型 $u_t^{\text{target}}(x\mid \varnothing)$

这种“把条件模型和无条件模型合并到一个网络中，再在推理时强化条件作用”的方法，就叫做 classifier-free guidance（CFG）。

### 注释 26（对一般概率路径的推广）

这里要注意：

$$
\tilde u_t(x\mid y)=(1-w)u_t^{\text{target}}(x)+w u_t^{\text{target}}(x\mid y)
$$

这个构造并不只适用于高斯概率路径，对任意概率路径都同样成立。

当

$$
w=1
$$

时，显然有

$$
\tilde u_t(x\mid y)=u_t^{\text{target}}(x\mid y).
$$

前面用高斯路径来推导，只是为了帮助理解这个构造背后的直觉，尤其是“放大一个假想 classifier 的贡献”这件事。

### 训练与 Classifier-Free Guidance

接下来，我们需要把式 (58) 中的 guided conditional flow matching 目标修改一下，以考虑

$$
y=\varnothing
$$

的情况。

问题在于：如果我们从真实数据分布中采样

$$
(z,y)\sim p_{\text{data}},
$$

那么永远不会自然地采到 $y=\varnothing$。因此，我们必须人为引入这种可能性。

做法是定义一个超参数 $\eta$，表示“丢弃原始标签 $y$ 并把它替换成 $\varnothing$ 的概率”。

于是得到 CFG 版 conditional flow matching 训练目标：

$$
L_{\mathrm{CFM}}^{\mathrm{CFG}}(\theta)
=
\mathbb{E}_{\square}
\bigl[
\|u_t^\theta(x\mid y)-u_t^{\text{target}}(x\mid z)\|^2
\bigr],
\tag{63}
$$

其中

$$
\square
=
(z,y)\sim p_{\text{data}}(z,y),\ 
t\sim \mathrm{Unif}[0,1],\ 
x\sim p_t(\cdot\mid z),
\ \text{并以概率 }\eta\text{把 }y\text{ 替换为 }\varnothing.
\tag{64}
$$

### 算法 5：高斯概率路径下的 Classifier-Free Guidance 训练

输入：成对数据集 $(z,y)\sim p_{\text{data}}$，神经网络 $u_t^\theta$

1. 对每个 mini-batch：
2. 从数据集中采样一个样本对 $(z,y)$
3. 采样随机时间

$$
t\sim \mathrm{Unif}[0,1]
$$

4. 采样噪声

$$
\epsilon\sim \mathcal{N}(0,I_d)
$$

5. 构造

$$
x=\alpha_t z+\beta_t\epsilon
$$

6. 以概率 $p$ 丢弃标签，即设置

$$
y\leftarrow \varnothing
$$

7. 计算损失

$$
L(\theta)=\|u_t^\theta(x\mid y)-(\dot\alpha_t z+\dot\beta_t\epsilon)\|^2
$$

8. 用梯度下降更新参数

### 小结 27（Flow Model 的 Classifier-Free Guidance）

给定：

- 无引导边缘向量场 $u_t^{\text{target}}(x\mid \varnothing)$
- 有引导边缘向量场 $u_t^{\text{target}}(x\mid y)$
- 引导强度 $w>1$

我们定义 classifier-free guidance 后的向量场为

$$
\tilde u_t(x\mid y)
=
(1-w)u_t^{\text{target}}(x\mid \varnothing)+w u_t^{\text{target}}(x\mid y).
\tag{65}
$$

通过让同一个神经网络同时逼近 $u_t^{\text{target}}(x\mid \varnothing)$ 和 $u_t^{\text{target}}(x\mid y)$，我们可以使用下面的 CFG-CFM 目标：

$$
L_{\mathrm{CFM}}^{\mathrm{CFG}}(\theta)
=
\mathbb{E}_{\square}
\bigl[
\|u_t^\theta(x\mid y)-u_t^{\text{target}}(x\mid z)\|^2
\bigr],
\tag{66}
$$

其中

$$
\square
=
(z,y)\sim p_{\text{data}}(z,y),\ 
t\sim \mathrm{Unif}[0,1],\ 
x\sim p_t(\cdot\mid z),
\ \text{并以概率 }\eta\text{把 }y\text{ 替换成 }\varnothing.
\tag{67}
$$

用更直白的话说，这个训练过程可以理解为：

1. 从真实数据分布中采样一个 $(z,y)$
2. 在 $[0,1)$ 上均匀采样一个时间 $t$
3. 从条件概率路径 $p_t(x\mid z)$ 中采样一个 $x$
4. 以概率 $\eta$ 把标签 $y$ 替换成 $\varnothing$
5. 让模型回归到条件向量场 $u_t^{\text{target}}(x\mid z)$

在推理时，对于固定的 $y$，我们可以这样采样：

- 初始化：

$$
X_0\sim p_{\text{init}}(x)
$$

- 模拟：

$$
dX_t = \tilde u_t^\theta(X_t\mid y)\,dt
$$

- 输出：

$$
X_1
$$

也就是说：

- 从一个简单分布（例如高斯分布）开始；
- 从 $t=0$ 模拟 ODE 到 $t=1$；
- 希望最终样本 $X_1$ 更加贴合条件变量 $y$。

需要注意的是：如果我们使用

$$
w>1,
$$

那么 $X_1$ 的分布就不一定仍然严格等于

$$
p_{\text{data}}(\cdot\mid y).
$$

也就是说，CFG 本质上是一个 heuristic，而不是一个严格保持目标分布不变的构造。

但经验上，它几乎总能显著提升样本和条件变量之间的对齐程度。事实上，几乎你看到的所有 AI 生成图像或视频，都会大量依赖 classifier-free guidance，而且常常使用

$$
w\ge 4.
$$

### 注释 28（扩展到 Diffusion 模型）

把上面的讨论从 flow model 扩展到 diffusion model 非常直接：

只需要把

$$
u_t^\theta(x\mid y)
$$

替换成

$$
\tilde u_t^\theta(x\mid y),
$$

然后像第 4 节那样使用 SDE 进行采样即可。
