# 第 7 章 离散扩散模型：用扩散构建语言模型

对应原讲义 `lecture_notes.pdf` 第 54-65 页。下面是这一章的中文翻译版；公式与算法已按 Markdown/LaTeX 形式整理，个别版式做了轻微重排，但内容保持原意。

## 7 离散扩散模型：用扩散构建语言模型

在前面的章节中，我们将流模型和扩散模型看作定义在欧氏空间 $\mathbb{R}^d$ 上的生成模型，它们能够生成由向量 $z\in\mathbb{R}^d$ 表示的数据点。然而，并不是所有数据都天然适合建模为欧氏空间 $\mathbb{R}^d$ 中的点。许多数据类型，例如文本或 DNA，更自然地应被看作离散状态空间 $\mathcal{S}$ 中的元素。最重要的是，语言由一串离散 token 组成，而这正是我们想建模的对象。

那么，我们如何把流模型和扩散模型应用到这类数据上呢？事实证明，我们在前面章节中学到的原理同样可以推广到这类数据。机器学习文献中把得到的模型称为离散扩散模型（discrete diffusion models）[5, 16]。不过需要记住的一点是：这里并不存在数学意义上的“扩散过程”本身，因为在离散状态空间上并不存在 SDE。我们不再使用 ODE/SDE，而是使用连续时间马尔可夫链（continuous-time Markov chains, CTMCs）。

接下来，我们将介绍 CTMC 模型（见 7.1 节）以及如何学习它们（见 7.2 节），从而用流模型和扩散模型的思想构建大语言模型（LLMs）。

## 7.1 连续时间马尔可夫链（CTMC）模型

这一节我们解释连续时间马尔可夫链（CTMC）。你可以把 CTMC 看作 SDE 在离散空间中的对应物；它允许我们构建能够生成离散状态的神经网络模型。进一步地，我们会介绍 CTMC 模型，也就是能够借助 CTMC 生成文本这类离散序列的神经网络模型。

我们先来刻画状态空间 $\mathcal{S}$。

令

$$
\mathcal{V}=\{v_1,\ldots,v_V\}
$$

表示词表（vocabulary）。状态空间定义为

$$
\mathcal{S}=\mathcal{V}^d,
$$

其中 $d\in\mathbb{N}$ 表示序列长度，$V\in\mathbb{N}$ 表示词表大小。对于语言来说，$\{v_1,\ldots,v_V\}$ 可以表示字母表或某个离散 token 集，而 $\mathcal{S}$ 就表示所有长度为 $d$ 的序列（或句子）集合。对于 DNA，$\{v_1,\ldots,v_V\}$ 可以是 4 种碱基，而 $\mathcal{S}$ 则是所有长度为 $d$ 的 DNA 序列。

接着，令 $X_t$ 是定义在 $\mathcal{S}$ 上的随机过程，也就是说它是一条随机轨迹

$$
X:[0,1]\to \mathcal{S},\quad t\mapsto X_t.
$$

我们要求 $X_t$ 是马尔可夫过程，也就是“无记忆”的过程。具体来说，它满足

$$
p(X_{t+h}\mid X_t,X_{t_1},\ldots,X_{t_k})
=
p(X_{t+h}\mid X_t),
$$

其中对任意 $0<h$ 和 $0\le t_1<t_2<\cdots<t_k<t$ 都成立。

换句话说，未来事件的概率只依赖于当前状态，过去的信息对未来不再有影响。注意，虽然 ODE/SDE 并不定义在离散状态空间上，但它们同样也是马尔可夫过程。这里的 $X_t$ 处在离散空间中，因此称为马尔可夫链，更准确地说，是连续时间马尔可夫链（CTMC）。

量

$$
p_{t+h\mid t}(X_{t+h}\mid X_t)
$$

就是转移概率。它们与初始分布 $X_0\sim p_0$ 一起，完全决定了这个 CTMC。因此，当我们说一个 CTMC 时，你也可以简单把它理解为一组转移概率 $p_{t+h\mid t}(X_{t+h}\mid X_t)$。

接下来，我们推导离散情形下“向量场”的对应物。由于我们现在处于离散空间中，状态之间只能发生跳跃（jump）或切换（switch），而不能像 ODE 那样沿某个连续方向运动。因此，我们定义速率矩阵（rate matrix）$Q_t(y\mid x)$，它刻画了从状态 $x\in\mathcal{S}$ 跳到状态 $y\in\mathcal{S}$ 的速率。

形式上，速率矩阵 $Q_t$ 是如下有界函数（并且对时间连续）：

$$
Q:\mathcal{S}\times \mathcal{S}\times [0,1]\to \mathbb{R},\qquad
(x,y,t)\mapsto Q_t(y\mid x)
\tag{84}
$$

其中 $Q_t(y\mid x)$ 描述从 $x$ 切换到 $y$ 的速率，并满足：

1. 对角线外的跳转速率非负：当 $x\neq y$ 时，

$$
Q_t(y\mid x)\ge 0
\tag{85}
$$

2. 留在原状态的速率等于离开该状态的总速率的相反数：

$$
Q_t(x\mid x)= -\sum_{y\neq x} Q_t(y\mid x)\qquad \text{对任意 }x
\tag{86}
$$

这两个条件都很直观。第一条说明，从 $x$ 跳到另一个状态 $y\neq x$ 的速率只能是非负的；“不发生跳转”对应速率为 0，因此出现小于 0 的跳转速率是没有意义的。第二条说明，留在 $x$ 的速率 $Q_t(x\mid x)$ 必须与离开 $x$ 的总速率相抵消；它本质上是一个一致性条件，表示系统要么留在 $x$，要么离开 $x$，没有第三种选择。

特别地，这也意味着 $Q_t(x\mid x)\le 0$。因此，$Q_t(y\mid x)$ 可以看作一个矩阵：其对角元素都非正，而非对角元素都非负。

现在我们可以定义离散版本的“微分方程”，也就是要求一个 CTMC “遵循”给定速率矩阵的条件。直觉上讲，我们希望 $X_t$ 的分布演化由速率矩阵 $Q_t$ 控制。换句话说，我们要求转移概率满足

$$
\left.\frac{d}{dh}p_{t+h\mid t}(X_{t+h}=y\mid X_t=x)\right|_{h=0}
=Q_t(y\mid x),
\qquad \forall x,y\in\mathcal{S},\ 0\le t.
\tag{87}
$$

左边是从 $x$ 跳到 $y$ 的**概率**在无穷小时间尺度上的**变化率**。我们要求这个变化率正好由速率矩阵给出。

我们可以快速检查这个要求是否合理。若直接把 $Q_t(y\mid x)$ 定义为式 (87) 右边，那它会是一个合法的速率矩阵吗？对于 $h=0$，从 $x$ 跳到任意不同状态 $y\neq x$ 的概率为 0，也即

$$
p_{t\mid t}(y\mid x)=0,\qquad y\neq x.
$$

因此其导数必须非负，于是当 $y\neq x$ 时有 $Q_t(y\mid x)\ge 0$，这验证了式 (85) 的第一条。进一步，

$$
\sum_{y\neq x} Q_t(y\mid x)
=
\sum_{y\neq x}
\left.\frac{d}{dh}p(X_{t+h}=y\mid X_t=x)\right|_{h=0}
=
\left.\frac{d}{dh}\Bigl(1-p(X_{t+h}=x\mid X_t=x)\Bigr)\right|_{h=0}
=-Q_t(x\mid x),
$$

这里用到了概率和为 1。于是式 (86) 也成立。这说明每一个 CTMC 至少都对应一个满足式 (87) 的速率矩阵。

那么反过来呢？如果我们先指定了一个 $Q_t$，是否存在对应的 CTMC？如果存在，它是否唯一？答案是肯定的。

### 定理 33（CTMC 的存在性与唯一性）

对于任意一个速率矩阵 $Q_t$（其关于时间 $t$ 有界且连续），都存在唯一一个马尔可夫链 $X_t$（也即唯一的一组转移概率 $p_{t+h\mid t}(y\mid x)$），使得式 (87) 成立。

对于感兴趣的读者，我们在附录 C 中给出了一个自包含证明。这个定理的关键意义在于：对于机器学习应用，只要我们构造出一个速率矩阵 $Q_t$（例如由神经网络给出），就可以假设存在唯一一个与之对应的马尔可夫链。

### 例 34（双状态且等跳转速率的 CTMC）

令 $\mathcal{S}=\{a,b\}$，考虑一个时间齐次 CTMC $(X_t)_{t\ge 0}$，它以常数速率 $\lambda>0$ 在两个状态之间切换：

$$
Q=
\begin{pmatrix}
-\lambda & \lambda\\
\lambda & -\lambda
\end{pmatrix}.
$$

则长度为 $h\ge 0$ 的时间增量上的转移概率也与时间 $t$ 无关，并且为

$$
\begin{pmatrix}
p(X_{t+h}=a\mid X_t=a) & p(X_{t+h}=a\mid X_t=b)\\
p(X_{t+h}=b\mid X_t=a) & p(X_{t+h}=b\mid X_t=b)
\end{pmatrix}
=
\frac12
\begin{pmatrix}
1+e^{-2\lambda h} & 1-e^{-2\lambda h}\\
1-e^{-2\lambda h} & 1+e^{-2\lambda h}
\end{pmatrix}.
$$

你可以手动检查式 (87) 确实成立，因此上述转移概率就是该速率矩阵对应的正确结果。这个速率非常直观：链会以瞬时速率 $\lambda$ 不断翻转。指数项 $e^{-2\lambda h}$ 描述了系统对初始状态的“记忆”如何衰减。

当时间趋于无穷大，即 $h\to\infty$ 时，有

$$
P(h)\to
\begin{pmatrix}
1/2 & 1/2\\
1/2 & 1/2
\end{pmatrix},
$$

因此这个链会逐渐忘记初始状态，最终停留在 $a$ 或 $b$ 的概率都变成 $1/2$。并且，切换速率 $\lambda$ 越大，这种收敛越快。

### CTMC 的模拟

接下来我们讨论如何模拟一条 CTMC 轨迹。令 $h>0$ 为步长，$p_{\text{init}}$ 为 $\mathcal{S}$ 上的初始分布，例如均匀分布 $p_{\text{init}}=\mathrm{Unif}_{\mathcal{S}}$。那么我们可以迭代地模拟：

$$
X_0\sim p_{\text{init}},\qquad
X_{t+h}\sim p_{t+h\mid t}(\cdot\mid X_t).
$$

如果我们知道 $p_{t+h\mid t}(\cdot\mid X_t)$，那事情就很简单。但对于除了最简单情形之外的大多数 CTMC，我们通常并不知道这个转移核的闭式表达，而只有速率矩阵 $Q_t$。不过，由式 (87) 可得：

$$
p_{t+h\mid t}(X_{t+h}=y\mid X_t=x)
=
p_{t\mid t}(X_t=y\mid X_t=x) + hQ_t(y\mid x)+R_t(h)
=
\mathbf{1}_{y=x}+hQ_t(y\mid x)+R_t(h),
$$

其中 $R_t(h)$ 是一个误差项，当 $h$ 很小时可以忽略。因此，当 $h$ 很小时，可以用下面的近似：

$$
p_{t+h\mid t}(X_{t+h}=y\mid X_t=x)
\approx
\mathbf{1}_{y=x}+hQ_t(y\mid x)
 =:\tilde p_{t+h\mid t}(y\mid x).
$$

因为速率矩阵满足前面给出的条件，所以对足够小的 $h$，$\tilde p_{t+h\mid t}(y\mid x)$ 的确是一个合法的概率分布。因此，我们可以近似地采样下一步：

$$
X_{t+h}\sim \tilde p_{t+h\mid t}(\cdot\mid x)
=
\bigl(\mathbf{1}_{y=x}+hQ_t(y\mid x)\bigr)_{y\in\mathcal{S}}.
\tag{88}
$$

由于上式只是一个离散分布，我们可以用标准方法直接采样。这就给出了一种简单的 CTMC 模拟方法。

### CTMC 模型

接下来，我们定义如何用神经网络对 CTMC 进行参数化。一个 CTMC 模型（或离散扩散模型）由初始分布 $p_{\text{init}}$ 和一个带参数 $\theta$ 的神经网络 $Q_t^\theta$ 构成，使得对每个输入 $x\in\mathcal{S}$，模型输出速率矩阵的一整列：

$$
x\mapsto \{Q_t^\theta(y\mid x)\}_{y\in\mathcal{S}}.
$$

之所以需要输出整列，是因为在模拟 CTMC 时（见式 (88)），我们需要这整列来决定下一状态的采样分布。

但这里有一个问题：状态空间 $\mathcal{S}$ 可能非常大。特别地，

$$
|\mathcal{S}|=V^d,
$$

其中 $V$ 是词表大小，$d$ 是序列长度。这种指数级增长使得我们几乎不可能在内存中存储速率矩阵的一整列；集合 $\{Q_t^\theta(y\mid x)\}_{y\in\mathcal{S}}$ 大到无法直接表示。

因此，我们必须对模型施加约束。具体来说，几乎所有 CTMC 模型都是因子化的（factorized；见图 18），这本质上是一种稀疏性约束。更准确地说，一个因子化 CTMC 模型是指：对任意

$$
y=(y_1,\ldots,y_d),\quad x=(x_1,\ldots,x_d)\in \mathcal{S}=\mathcal{V}^d,
$$

都满足

$$
Q_t^\theta(y\mid x)=0,
\qquad
\text{只要 } y_i\neq x_i \text{ 的位置超过一个。}
$$

我们把与 $x$ 至多只在一个 token 上不同的所有 $y$ 称为 $x$ 的邻居，记作 $\mathcal{N}(x)$。于是这样的因子化 CTMC 模型可写为

$$
x\mapsto \{Q_t^\theta(y\mid x)\}_{y\in \mathcal{N}(x)}
=
\begin{pmatrix}
Q_t^\theta(v_1,1\mid x) & \cdots & Q_t^\theta(v_V,1\mid x)\\
\vdots & \ddots & \vdots\\
Q_t^\theta(v_1,d\mid x) & \cdots & Q_t^\theta(v_V,d\mid x)
\end{pmatrix}.
$$

这里 $Q_t^\theta(y\mid x)=Q_t^\theta(v_i,j\mid x)$ 表示从

$$
x=(x_1,\ldots,x_d)
$$

跳到某个邻居 $y$ 的速率，其中这个邻居是把第 $j$ 个位置替换为 $v_i$ 得到的：

$$
y=(x_1,\ldots,x_{j-1},v_i,x_{j+1},\ldots,x_d).
$$

每一行都对应一个位置的速率矩阵，因此我们要求

$$
Q_t^\theta(v,i\mid x)\ge 0 \quad \text{当 } v\neq x_i,
$$

以及

$$
Q_t^\theta(x_i,i\mid x)
=
-\sum_{v\neq x_i} Q_t^\theta(v,i\mid x).
$$

这些条件都很容易通过神经网络输出加以保证。例如，可以使用长度为 $d$、输出维度为 $V$ 的 Transformer。注意，因子化后模型输出的形状是 $d\times V$，其大小只随维度线性增长，而不是指数增长。

### CTMC 模型的采样

要从一个 CTMC 模型中采样，我们先从 $X_0\sim p_{\text{init}}$ 开始，然后不断按式 (88) 迭代采样下一状态。算法 7 给出了具体流程。对于因子化 CTMC 模型，可以使用并行的逐 token Euler 近似：在一个很小的步长 $h>0$ 内，各个 token 独立更新。这个近似在 $h$ 的一阶上与完整 CTMC 的 Euler 步一致，但会以 $O(h^2)$ 的概率出现多个 token 同时更新。

### 算法 7：从因子化 CTMC 模型中采样

输入：因子化速率网络 $Q_t^\theta$、初始分布 $p_{\text{init}}$、步数 $n$

1. 令 $t\leftarrow 0$，步长 $h\leftarrow 1/n$。
2. 采样 $X_0\sim p_{\text{init}}$，其中 $X_0=(X_0^{(1)},\ldots,X_0^{(d)})\in \mathcal{V}^d$。
3. 对 $i=1,\ldots,n$：
4. 计算因子化跳转速率 $\{q_j(v)\}_{j=1,\ldots,d,\ v\in\mathcal{V}}\leftarrow Q_t^\theta(\cdot\mid X_t)$。
5. 对每个位置 $j=1,\ldots,d$（可并行）：
6. 令 $x\leftarrow X_t^{(j)}$，即当前位置的当前 token。
7. 定义单位置 Euler 转移概率

$$
\tilde p_{j,t}(v\mid x)=
\begin{cases}
hq_j(v), & v\neq x,\\
1-h\sum_{v'\in \mathcal{V}\setminus \{x\}} q_j(v'), & v=x.
\end{cases}
$$

8. 采样 $X_{t+h}^{(j)}\sim \mathrm{Categorical}(\{\tilde p_{j,t}(v\mid x)\}_{v\in\mathcal{V}})$。
9. 所有位置完成后，令 $t\leftarrow t+h$。
10. 返回 $X_1$。

## 7.2 CTMC 模型的训练

下面我们讨论如何学习 CTMC 模型。它的思路与 flow matching 是一致的：

1. 构造一条连接噪声与数据的概率路径；
2. 推导条件速率矩阵和边缘速率矩阵；
3. 以无模拟（simulation-free）的方式学习边缘速率矩阵。

下面我们一步一步展开说明。

在这一节中，数据分布 $p_{\text{data}}$ 是定义在 $\mathcal{S}$ 上的一个分布，可由概率质量函数表示：

$$
p_{\text{data}}:\mathcal{S}\to \mathbb{R}_{\ge 0},\qquad
z\mapsto p_{\text{data}}(z),
$$

并满足

$$
\sum_{z\in\mathcal{S}} p_{\text{data}}(z)=1.
$$

我们并不知道 $p_{\text{data}}$ 的解析形式，但在训练时可以访问从它采样得到的数据点 $z\sim p_{\text{data}}$，这些样本来自数据集，例如整个互联网中的文本。我们的目标是学会生成服从 $p_{\text{data}}$ 的样本。

也就是说，我们要训练 CTMC 模型 $Q_t^\theta$，使得

$$
X_0\sim p_{\text{init}},\quad
X_t \text{ 是由 }Q_t^\theta \text{ 诱导的 CTMC}
\quad \Longrightarrow \quad
X_1\sim p_{\text{data}}.
$$

你会发现，这与欧氏空间情形（见第 2、3 节）本质上没有区别，只不过这里我们用的是 CTMC 模型，而不是流模型/扩散模型。

### 7.2.1 条件概率路径与边缘概率路径

定义 $\delta_z(x)$ 为如下函数：

$$
\delta_z(x)=
\begin{cases}
1, & x=z,\\
0, & x\neq z.
\end{cases}
$$

一个离散的条件概率路径是指一族分布 $p_t(x\mid z)$，其中 $x,z\in\mathcal{S}$ 且 $0\le t\le 1$，满足

$$
p_0(\cdot\mid z)=p_{\text{init}},\qquad
p_1(\cdot\mid z)=\delta_z.
$$

因此，和欧氏空间中的情形类似，离散条件概率路径是在一个与 $z$ 无关的分布与一个全部质量都集中在 $z$ 上的分布之间进行插值。

对应的离散边缘概率路径定义为

$$
p_t(x)=\sum_{z\in\mathcal{S}} p_t(x\mid z)p_{\text{data}}(z).
$$

很容易验证，边缘概率路径确实在“噪声”和“数据”之间插值：

$$
p_0=p_{\text{init}},\qquad p_1=p_{\text{data}}.
\tag{89}
$$

### 例 35（因子化混合路径：每个 token 独立加噪）

令 $\mathcal{S}=\mathcal{V}^d$，并设

$$
p_{\text{init}}(x)=\prod_{j=1}^d p_{\text{init}}^{(j)}(x_j)
$$

是一个因子化初始分布。固定一个调度函数（scheduler）$0\le \kappa_t\le 1$，满足 $\kappa_0=0,\ \kappa_1=1$，且

$$
\frac{d}{dt}\kappa_t=\dot\kappa_t\ge 0.
$$

定义条件路径为

$$
p_t(x\mid z)
=
\prod_{j=1}^d
\Bigl((1-\kappa_t)\,p_{\text{init}}^{(j)}(x_j)+\kappa_t\,\delta_{z_j}(x_j)\Bigr).
$$

等价地，采样 $x\sim p_t(\cdot\mid z)$ 可以按如下方式进行：独立采样 mask $m_j\in\{0,1\}$ 以及噪声 $\xi_j\sim p_{\text{init}}^{(j)}$，然后设

$$
m_j\sim \mathrm{Bernoulli}(\kappa_t),\qquad
\xi_j\sim p_{\text{init}}^{(j)},
$$

并令

$$
x_j = m_j z_j + (1-m_j)\xi_j,\qquad
x=(x_1,\ldots,x_d),
\qquad j=1,\ldots,d.
$$

上面这条路径称为因子化混合路径（factorized mixture path）。这个过程本质上是在序列的每个位置上，以概率 $1-\kappa_t$ 独立地“破坏”第 $j$ 个 token。也就是说，当 $t=0$ 时，$1-\kappa_t=1$，所有信息都被摧毁；当 $t=1$ 时，$1-\kappa_t=0$，信息不再被破坏。

这与高斯概率路径（例 8）相似，都是以由 $\kappa_t$ 决定的速度逐步抹除信息；但也有所不同：因子化混合路径并不会搬运概率质量（因为在离散空间中不存在连续“方向”），它只是逐渐淡出一个分布并淡入另一个分布。

### 7.2.2 条件速率矩阵与边缘速率矩阵

接下来我们构造离散 flow matching 的训练目标。首先，我们构造条件速率矩阵，它对应于 flow matching 中的条件向量场。

对每个数据点 $z\in\mathcal{S}$，令 $Q_t^z(y\mid x)$ 是一个速率矩阵。如果它满足

$$
X_0\sim p_{\text{init}},\qquad
X_t \text{ 是由 }Q_t^z\text{ 诱导的 CTMC}
\quad \Longrightarrow \quad
X_t\sim p_t(\cdot\mid z),
$$

那么我们就称它为条件速率矩阵。

也就是说，条件速率矩阵所对应的 CTMC 会“沿着”条件概率路径演化。条件速率矩阵是构造边缘速率矩阵的基础，而边缘速率矩阵则沿着边缘概率路径演化。

### 定理 36（离散边缘化技巧）

定义边缘速率矩阵

$$
Q_t(y\mid x)
=
\sum_{z\in\mathcal{S}} Q_t^z(y\mid x)\frac{p_t(x\mid z)p_{\text{data}}(z)}{p_t(x)}
=
\sum_{z\in\mathcal{S}} Q_t^z(y\mid x)\,p_{1\mid t}(z\mid x),
\tag{90}
$$

其中

$$
p_{1\mid t}(z\mid x):=\frac{p_t(x\mid z)p_{\text{data}}(z)}{p_t(x)}.
$$

则 $Q_t$ 是一个合法的速率矩阵，并满足

$$
X_0\sim p_{\text{init}},\qquad
X_t \text{ 是由 }Q_t\text{ 诱导的 CTMC}
\quad \Longrightarrow \quad
X_t\sim p_t.
$$

特别地，由式 (89) 可知 $X_1\sim p_{\text{data}}$，也就是说边缘速率矩阵对应的 CTMC 会把噪声转换成数据。

为了证明这个结论，我们需要 CTMC 的一个基本方程，即 Kolmogorov 前向方程。

### 命题 2（Kolmogorov 前向方程）

设对每个 $0\le t\le 1$ 都有一组定义在 $\mathcal{S}$ 上的分布 $p_t$。再设 $X_t$ 是一个速率矩阵为 $Q_t$、初始分布为 $p_0$ 的 CTMC。那么对所有 $0\le t\le 1$，$X_t\sim p_t$ 当且仅当满足 Kolmogorov 前向方程（KFE）：

$$
\frac{d}{dt}p_t(x)
=
\sum_{y\in\mathcal{S}} Q_t(x\mid y)p_t(y).
$$

#### KFE 的证明

先证明必要性。假设 $p_t(x)$ 确实是 CTMC 的真实边缘分布，即对每个 $0\le t\le 1$ 都有 $X_t\sim p_t$。则

$$
\frac{d}{dt}p_t(x)
=
\left.\frac{d}{dh}p_{t+h}(x)\right|_{h=0}
=
\left.\frac{d}{dh}\sum_y p_{t+h\mid t}(x\mid y)p_t(y)\right|_{h=0}
=
\sum_y \left.\frac{d}{dh}p_{t+h\mid t}(x\mid y)\right|_{h=0}p_t(y)
=
\sum_y Q_t(x\mid y)p_t(y).
$$

于是 KFE 成立。

再证明充分性。把 KFE 写成矩阵形式：

$$
\frac{d}{dt}p_t = Q_t p_t,
$$

其中把 $p_t=(p_t(x))_{x\in\mathcal{S}}$ 看作向量，把 $Q_t=(Q_t(y\mid x))_{x,y\in\mathcal{S}}$ 看作矩阵。这个方程是向量空间 $\mathbb{R}^{\mathcal{S}}$ 上的一个线性 ODE，而它的初始条件由 $p_0$ 给定。因此，如果另一组边缘分布 $q_t$ 也满足这个方程，那么由 ODE 的唯一性（见定理 3）可知 $q_t=p_t$。这就说明 KFE 也是充分条件。

#### 定理 36 的证明

利用 KFE，只需证明按定理定义的边缘速率矩阵满足 KFE：

$$
\frac{d}{dt}p_t(x)
=
\frac{d}{dt}\sum_{z\in\mathcal{S}} p_t(x\mid z)p_{\text{data}}(z)
=
\sum_{z\in\mathcal{S}} \frac{d}{dt}p_t(x\mid z)p_{\text{data}}(z)
$$

$$
=
\sum_{z\in\mathcal{S}}
\left[
\sum_{y\in\mathcal{S}}Q_t^z(x\mid y)p_t(y\mid z)
\right]p_{\text{data}}(z)
$$

$$
=
\sum_{y\in\mathcal{S}} p_t(y)
\sum_{z\in\mathcal{S}}Q_t^z(x\mid y)\frac{p_t(y\mid z)p_{\text{data}}(z)}{p_t(y)}
=
\sum_{y\in\mathcal{S}} p_t(y)Q_t(x\mid y).
$$

因此 KFE 成立，由命题 2 可得结论。

接下来，我们为因子化混合路径推导一个具体的条件速率矩阵。

### 例 37（因子化混合路径的条件速率矩阵）

记

$$
\frac{d}{dt}\kappa_t=\dot\kappa_t.
$$

则因子化混合路径对应的条件速率矩阵也是因子化的，并且满足

$$
Q_t^z(y\mid x)=\bigl(Q_t^z(v_i,j\mid x_j)\bigr)_{v_i,j},
$$

其中

$$
Q_t^z(v_i,j\mid x_j)
=
\frac{\dot\kappa_t}{1-\kappa_t}\bigl(\delta_{z_j}(v_i)-\delta_{x_j}(v_i)\bigr).
$$

等价地，它可以写成分段形式：

$$
Q_t^z(v_i,j\mid x_j)=
\begin{cases}
0, & x_j=z_j,\\
\dfrac{\dot\kappa_t}{1-\kappa_t}, & v_i=z_j,\ x_j\neq z_j,\\
0, & v_i\neq z_j,\ v_i\neq x_j,\ x_j\neq z_j,\\
-\dfrac{\dot\kappa_t}{1-\kappa_t}, & v_i=x_j,\ x_j\neq z_j.
\end{cases}
$$

注意，这是一个非常简单的速率矩阵：它只允许跳向 $z_j$。也就是说，如果某个位置 $j$ 的 token 被更新，那么它一定跳到终点数据 $z=(z_1,\ldots,z_d)$ 在该位置的 token 值；并且只有在当前位置还没有等于 $z_j$ 时才会发生这类跳转。

#### 证明

注意到因子化混合路径完全分解为彼此独立的分量，而建议的条件速率矩阵也是如此。因此不失一般性，可以假设 $d=1$，也就是逐维计算。

于是有

$$
\frac{d}{dt}p_t(x\mid z)
=
\frac{d}{dt}\Bigl[(1-\kappa_t)p_{\text{init}}(x)+\kappa_t\delta_z(x)\Bigr]
$$

$$
=
\dot\kappa_t\delta_z(x)-\dot\kappa_t p_{\text{init}}(x)
$$

$$
=
\frac{\dot\kappa_t}{1-\kappa_t}
\Bigl(\delta_z(x)-\bigl[(1-\kappa_t)p_{\text{init}}(x)+\kappa_t\delta_z(x)\bigr]\Bigr)
$$

$$
=
\frac{\dot\kappa_t}{1-\kappa_t}\bigl(\delta_z(x)-p_t(x\mid z)\bigr).
$$

进一步可改写为

$$
\frac{d}{dt}p_t(x\mid z)
=
\sum_{y\neq x}\frac{\dot\kappa_t}{1-\kappa_t}\delta_z(x)p_t(y\mid z)
\;+\;
\frac{\dot\kappa_t}{1-\kappa_t}\bigl(\delta_z(x)-1\bigr)p_t(x\mid z)
$$

$$
=
\sum_{y\in\mathcal{S}} Q_t^z(x\mid y)p_t(y\mid z).
$$

因此 KFE 成立，命题得证。

### 7.2.3 学习边缘速率矩阵

现在我们来推导训练 CTMC 模型的核心算法。由定理 36 可知，要训练一个 CTMC 模型 $Q_t^\theta(y\mid x)$，只需要学习对应的边缘速率矩阵。

从这一节开始，我们只考虑因子化混合路径（见例 35），因为这是目前大多数离散扩散/离散 flow matching 模型使用的路径。在这种情况下，边缘速率矩阵具有非常直观的形式。

### 定理 38（因子化混合路径的边缘化技巧）

因子化混合路径的边缘速率矩阵也是因子化的，并且满足

$$
Q_t(v_i,j\mid x)
=
\frac{\dot\kappa_t}{1-\kappa_t}
\Bigl(p_{1\mid t}(z_j=v_i\mid x)-\delta_{x_j}(v_i)\Bigr),
$$

其中 $p_{1\mid t}(z_j=v_i\mid x)$ 表示：在给定完整噪声序列 $x$ 的条件下，第 $j$ 个位置（即序列中第 $j$ 个 token）等于 $v_i$ 的条件概率。

#### 证明

边缘速率矩阵定义为

$$
Q_t(y\mid x)=\sum_{z\in\mathcal{S}} Q_t^z(y\mid x)p_{1\mid t}(z\mid x).
\tag{91}
$$

当 $y$ 与 $x$ 不是邻居（也就是超过一个 token 不同）时，对任意 $z$ 都有 $Q_t^z(y\mid x)=0$。因此这种情况下 $Q_t(y\mid x)=0$，这说明边缘速率矩阵同样也是因子化的。

于是，

$$
Q_t(v_i,j\mid x)
=
\sum_{z\in\mathcal{S}} Q_t^z(v_i,j\mid x)p_{1\mid t}(z\mid x)
\tag{92}
$$

$$
=
\sum_{z\in\mathcal{S}}
\frac{\dot\kappa_t}{1-\kappa_t}
\bigl(\delta_{z_j}(v_i)-\delta_{x_j}(v_i)\bigr)p_{1\mid t}(z\mid x)
\tag{93}
$$

$$
=
\frac{\dot\kappa_t}{1-\kappa_t}
\left(
\sum_{z\in\mathcal{S}}\delta_{z_j}(v_i)p_{1\mid t}(z\mid x)
-\delta_{x_j}(v_i)
\right)
\tag{94}
$$

$$
=
\frac{\dot\kappa_t}{1-\kappa_t}
\Bigl(p_{1\mid t}(z_j=v_i\mid x)-\delta_{x_j}(v_i)\Bigr).
\tag{95}
$$

证毕。

上一个定理非常关键：边缘速率矩阵本质上就是对概率

$$
p_{1\mid t}(z_j=v_i\mid x)
$$

的一种重参数化。这实际上就是在每个 token 位置 $j=1,\ldots,d$ 上学习一个分类器。

换句话说，我们只需定义一个“去噪概率网络”：

$$
p_{1\mid t}^\theta:
x
\longmapsto
\bigl(p_{1\mid t}^\theta(z_j=v_i\mid x)\bigr)_{j=1,\ldots,d,\ v_i\in \mathcal{V}}.
$$

注意，网络输出的形状是 $d\times V$。对每个 token 位置的概率，都可以通过一个简单的 softmax 层得到。网络本身可以是标准的 sequence-to-sequence 模型，例如 Transformer（见 6.1.2 节）。

由于这本质上就是每个位置 $j$ 上的一个分类问题，因此我们可以直接对每个 $j=1,\ldots,d$ 使用交叉熵损失进行训练。由此得到离散 Flow Matching 损失：

$$
L_{\mathrm{DFM}}(\theta)
=
\mathbb{E}_{z\sim p_{\text{data}},\ t\sim \mathrm{Unif}[0,1],\ x\sim p_t(\cdot\mid z)}
\left[
\sum_{j=1}^d -\log p_{1\mid t}^\theta(z_j\mid x)
\right].
$$

这非常值得注意：为了训练一个生成模型，我们所需要做的只是对每个位置 $j$ 训练一个分类器。正如连续 flow matching 被化简成一个简单的回归问题（见第 3 节）一样，离散 flow matching 和离散扩散模型也被化简成了简单的分类训练问题。算法 8 总结了训练过程。训练完成后，我们可以使用算法 7 进行采样。

### 例 39（Masked Diffusion Language Model）

上述方法的一个具体实例是 masked diffusion language model（MDLM）。其核心思想是：在原词表

$$
\mathcal{V}=\{v_1,\ldots,v_V\}
$$

中加入一个新的 token `[MASK]`，用于表示该位置被遮蔽（masked）或缺失。于是新的词表变成

$$
\mathcal{V}=\{v_1,\ldots,v_V,[\mathrm{MASK}]\}.
$$

初始点则直接设为全 mask 序列：

$$
[\mathrm{MASK}]^d.
$$

形式上，这等价于令

$$
p_{\text{init}}=\delta_{[\mathrm{MASK}]^d}.
$$

在这一框架下，采样过程就对应于从“全 mask”逐渐恢复出完整句子。

### 算法 8：训练因子化 CTMC 模型（离散扩散）

输入：

- 来自数据分布 $z\sim p_{\text{data}}$ 的序列数据集，其中 $z=(z_1,\ldots,z_d)\in \mathcal{V}^d$
- 初始（噪声）token 边缘分布 $p_{\text{init}}^{(j)}$，定义在 $\mathcal{V}$ 上
- 调度函数 $\kappa_t\in[0,1]$
- 后验网络 $f_\theta$，输出每个位置对 $\mathcal{V}$ 的 logits
- 优化器 `Opt`

训练步骤：

1. 对每次训练迭代：
2. 采样一个数据点 $z\sim p_{\text{data}}$。
3. 采样时间 $t\sim \mathrm{Unif}[0,1]$，并计算 $\kappa\leftarrow \kappa_t$。
4. 从因子化混合路径采样噪声状态 $x\sim p_t(\cdot\mid z)$。
5. 对每个位置 $j=1,\ldots,d$（可并行）：
6. 采样 mask：$m_j\sim \mathrm{Bernoulli}(\kappa)$。
7. 采样噪声 token：$\xi_j\sim p_{\text{init}}^{(j)}$。
8. 设置

$$
x_j \leftarrow m_j z_j + (1-m_j)\xi_j.
$$

9. 得到 $x=(x_1,\ldots,x_d)$。
10. 用网络预测终点 token 的后验分布：令

$$
\ell_j(\cdot)\leftarrow f_\theta(x,t)_j,
\qquad
p_{1\mid t}^\theta(v\mid x)_j=\mathrm{Softmax}\bigl(\ell_j(v)\bigr).
$$

11. 定义离散 Flow Matching 损失（对 $z$ 的逐 token 负对数似然）：

$$
L_{\mathrm{DFM}}(\theta)
\leftarrow
\sum_{j=1}^d -\log p_{1\mid t}^\theta(z_j\mid x)_j.
$$

12. 用优化器更新参数：

$$
\theta \leftarrow \mathrm{Opt.step}\,\nabla_\theta L_{\mathrm{DFM}}(\theta).
$$

### Masked Diffusion 轨迹示意

```text
t = 0     [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]
t = 0.25  [MASK] [MASK] [MASK] on     [MASK] [MASK] [MASK]
t = 0.75  [MASK] cat    [MASK] on     the    mat    [MASK]
t = 1     The    cat    sat    on     the    mat    .
```

这就完成了训练和采样 CTMC 模型的一整套流程，从而使我们能够生成文本这类离散序列。当前最先进的离散扩散模型 [4] 基本都在使用这里描述的这套方案，只是把神经网络（通常是 Transformer）扩展到了网络规模数据上进行训练。
