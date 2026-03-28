# 第 3 节 Flow Matching（中文翻译）

对应原讲义 `lecture_notes.pdf` 第 14-24 页，即 Section 3：`Flow Matching`，包含 3.1、3.2、3.3 三个小节。

## 3 Flow Matching

在上一节中，我们已经把 flow model 和 diffusion model 构造成了由神经网络向量场 $u_t^\theta$ 参数化的生成模型。不过，我们还没有讨论怎样训练它们，也就是怎样优化参数 $\theta$，使得生成模型能够输出“有意义”的内容，比如好看的图像、精彩的视频等。

接下来，我们介绍 flow matching [25, 1, 27]。它是一种训练 $u_t^\theta$ 的算法，特点是简单、可扩展，而且代表了当前最先进的方法之一。

在这一节里，我们先只关注 flow model。也就是说，我们有一个神经网络 $u_t^\theta$，并通过模拟下面的 ODE 来从生成模型中采样：

$$
X_0\sim p_{\text{init}},\qquad
\frac{d}{dt}X_t = u_t^\theta(X_t)
\qquad \text{(Flow model)}
\tag{10}
$$

然后把终点 $t=1$ 时的 $X_1$ 作为样本。

正如前面讨论过的，我们的目标是让 $X_1$ 的分布等于数据分布 $p_{\text{data}}$，也就是

$$
X_1\sim p_{\text{data}}.
$$

因此，“如何训练神经网络”本质上等价于下面这个问题：

> 如何优化参数 $\theta$，使得模拟式 (10) 的 flow model 后，得到的终点样本满足 $X_1\sim p_{\text{data}}$？

## 3.1 条件概率路径与边缘概率路径

Flow matching 的第一步，是指定一条概率路径（probability path）。

直观地说，概率路径描述了噪声分布 $p_{\text{init}}$ 与数据分布 $p_{\text{data}}$ 之间的一种渐进插值。为什么要这样做？回忆一下，我们希望 ODE 的轨迹满足：

- 当 $t=0$ 时，$X_0\sim p_{\text{init}}$
- 当 $t=1$ 时，$X_1\sim p_{\text{data}}$

但是对于中间时刻 $0<t<1$，轨迹上的分布应该是什么样子？事实证明，这一部分我们其实有一定自由度，而概率路径正是对“中间该如何演化”这一点的数学刻画。

在下面，对任意数据点 $z\in\mathbb{R}^d$，我们用 $\delta_z$ 表示 Dirac delta“分布”。它是最简单的分布：从 $\delta_z$ 采样总是返回 $z$ 本身，也就是说它是完全确定性的。

一个条件（插值）概率路径（conditional probability path），是一族定义在 $\mathbb{R}^d$ 上的分布 $p_t(x\mid z)$，满足：

$$
p_0(\cdot\mid z)=p_{\text{init}},\qquad
p_1(\cdot\mid z)=\delta_z,
\qquad \forall z\in\mathbb{R}^d.
\tag{11}
$$

换句话说，一条条件概率路径会逐渐地把初始分布 $p_{\text{init}}$ 变成单个数据点 $z$。你可以把概率路径理解成“分布空间中的一条轨迹”。

每一条条件概率路径 $p_t(x\mid z)$ 都会诱导出一条边缘概率路径（marginal probability path）$p_t(x)$。它的定义方式是：

1. 先从数据分布中采样一个数据点 $z\sim p_{\text{data}}$
2. 再从条件路径 $p_t(\cdot\mid z)$ 中采样

于是得到

$$
z\sim p_{\text{data}},\quad x\sim p_t(\cdot\mid z)
\quad\Longrightarrow\quad
x\sim p_t.
\tag{12}
$$

其密度写作

$$
p_t(x)=\int p_t(x\mid z)\,p_{\text{data}}(z)\,dz.
\tag{13}
$$

注意，我们知道如何从 $p_t$ 采样，但通常并不能高效计算它的密度值 $p_t(x)$，因为上面的积分往往是不可解的。也就是说，式 (12) 可用于采样，但式 (13) 往往无法直接计算。

你可以自己检查：由式 (11) 的定义可知，边缘概率路径 $p_t$ 确实在 $p_{\text{init}}$ 和 $p_{\text{data}}$ 之间插值：

$$
p_0=p_{\text{init}},\qquad p_1=p_{\text{data}}.
\tag{14}
$$

在各种概率路径里，最重要、也最常用的一类，就是高斯概率路径（Gaussian probability path）。这一点非常关键，所以讲义特别建议认真读下面这个例子。

### 例 8（高斯条件概率路径）

一种特别常用的概率路径是高斯概率路径。当前大多数最先进模型都在使用它。

设 $\alpha_t,\beta_t$ 是两个噪声调度函数（noise schedulers）：它们都是连续可微的单调函数，并满足

$$
\alpha_0=\beta_1=0,\qquad
\alpha_1=\beta_0=1.
$$

然后定义条件概率路径为

$$
p_t(\cdot\mid z)=\mathcal{N}(\alpha_t z,\beta_t^2 I_d)
\qquad\triangleright\ \text{高斯条件路径}
\tag{15}
$$

由我们对 $\alpha_t,\beta_t$ 的约束条件，可以得到

$$
p_0(\cdot\mid z)=\mathcal{N}(\alpha_0 z,\beta_0^2 I_d)=\mathcal{N}(0,I_d),
$$

以及

$$
p_1(\cdot\mid z)=\mathcal{N}(\alpha_1 z,\beta_1^2 I_d)=\delta_z.
$$

这里我们用到了这样一个事实：均值为 $z$、方差为 0 的高斯分布，正是 $\delta_z$。

因此，对 $p_{\text{init}}=\mathcal{N}(0,I_d)$ 而言，这个 $p_t(x\mid z)$ 的定义确实满足式 (11)，所以它是一条合法的条件插值路径。

从这条边缘路径采样，可以写成

$$
z\sim p_{\text{data}},\qquad
\epsilon\sim p_{\text{init}}=\mathcal{N}(0,I_d)
\quad\Longrightarrow\quad
x=\alpha_t z+\beta_t\epsilon \sim p_t.
\tag{16}
$$

直观上看，当 $t$ 越小时，我们加入的噪声越多；直到 $t=0$ 时，样本中只剩纯噪声。

## 3.2 条件向量场与边缘向量场

概率路径 $(p_t)_{0\le t\le 1}$ 只告诉我们：轨迹上的点 $X_t$ 希望服从什么分布，即

$$
X_t\sim p_t.
$$

但这目前还只是我们的“愿望”。真正的问题是：

> 我们怎样找到一个向量场，使得 ODE 的轨迹真的沿着这条概率路径演化？

Flow matching 正是在这一节中显式构造出这样一个向量场，它叫做边缘向量场（marginal vector field）。

对每个数据点 $z\in\mathbb{R}^d$，令 $u_t^{\text{target}}(\cdot\mid z)$ 表示一个条件向量场。它可以是任何一个向量场，只要它所对应的 ODE 能产生条件概率路径 $p_t(\cdot\mid z)$，也就是满足

$$
X_0\sim p_{\text{init}},\qquad
\frac{d}{dt}X_t = u_t^{\text{target}}(X_t\mid z)
\quad\Longrightarrow\quad
X_t\sim p_t(\cdot\mid z),\qquad 0\le t\le 1.
\tag{17}
$$

在很多情况下，我们可以手工推导出一个条件向量场 $u_t^{\text{target}}(\cdot\mid z)$。下面的例 10 就会对高斯概率路径做这个推导。

乍一看，条件向量场似乎没什么用，因为如果你知道终点数据点是 $z$，那对应 ODE 的终点就会坍缩到 $X_1=z$，也就是说你只是把已知数据点又“生成”了一遍。

但事实上，条件向量场是构造真正生成样本的向量场的基础。下面这个结果说明了如何从条件向量场得到能生成 $p_{\text{data}}$ 样本的边缘向量场。

### 定理 9（边缘化技巧）

设 $u_t^{\text{target}}(x\mid z)$ 是一个条件向量场（满足式 (17)）。定义边缘向量场

$$
u_t^{\text{target}}(x)
=
\int
u_t^{\text{target}}(x\mid z)
\frac{p_t(x\mid z)p_{\text{data}}(z)}{p_t(x)}
\,dz.
\tag{18}
$$

则这个边缘向量场会沿着边缘概率路径演化，也就是说

$$
X_0\sim p_{\text{init}},\qquad
\frac{d}{dt}X_t = u_t^{\text{target}}(X_t)
\quad\Longrightarrow\quad
X_t\sim p_t,\qquad 0\le t\le 1.
\tag{19}
$$

特别地，在 $t=1$ 时就有

$$
X_1\sim p_{\text{data}}.
$$

因此我们可以说：这个边缘向量场 $u_t^{\text{target}}$ 实现了“把噪声分布 $p_{\text{init}}$ 变成数据分布 $p_{\text{data}}$”。

### 例 10（高斯概率路径的目标 ODE）

继续考虑高斯概率路径

$$
p_t(\cdot\mid z)=\mathcal{N}(\alpha_t z,\beta_t^2 I_d),
$$

其中 $\alpha_t,\beta_t$ 是噪声调度函数。记

$$
\dot\alpha_t = \partial_t \alpha_t,\qquad
\dot\beta_t = \partial_t \beta_t
$$

表示它们对时间的导数。

这里我们要说明：下面这个条件高斯向量场

$$
u_t^{\text{target}}(x\mid z)
=
\left(
\dot\alpha_t-\frac{\dot\beta_t}{\beta_t}\alpha_t
\right)z
+
\frac{\dot\beta_t}{\beta_t}x
\tag{20}
$$

确实是一个合法的条件向量场。也就是说，如果初值满足

$$
X_0\sim \mathcal{N}(0,I_d),
$$

那么其 ODE 轨迹就满足

$$
X_t\sim p_t(\cdot\mid z)=\mathcal{N}(\alpha_t z,\beta_t^2 I_d).
$$

#### 证明

我们先构造一个条件 flow：

$$
\psi_t^{\text{target}}(x\mid z)=\alpha_t z+\beta_t x.
$$

若 $X_t$ 是这个 flow 对应的 ODE 轨迹，且

$$
X_0\sim p_{\text{init}}=\mathcal{N}(0,I_d),
$$

那么根据定义，

$$
X_t
=
\psi_t^{\text{target}}(X_0\mid z)
=
\alpha_t z+\beta_t X_0
\sim
\mathcal{N}(\alpha_t z,\beta_t^2 I_d)
=
p_t(\cdot\mid z).
\tag{21}
$$

因此它确实沿着条件概率路径演化。

接下来，我们需要从 $\psi_t^{\text{target}}(x\mid z)$ 中提取出向量场 $u_t^{\text{target}}(x\mid z)$。根据 flow 的定义（即式 (2b)），有

$$
\frac{d}{dt}\psi_t^{\text{target}}(x\mid z)
=
u_t^{\text{target}}(\psi_t^{\text{target}}(x\mid z)\mid z),
\qquad \forall x,z\in\mathbb{R}^d.
$$

把 $\psi_t^{\text{target}}(x\mid z)=\alpha_t z+\beta_t x$ 代入，可得

$$
\dot\alpha_t z+\dot\beta_t x
=
u_t^{\text{target}}(\alpha_t z+\beta_t x\mid z).
$$

再做变量代换

$$
x\mapsto \frac{x-\alpha_t z}{\beta_t},
$$

得到

$$
\dot\alpha_t z+\dot\beta_t\frac{x-\alpha_t z}{\beta_t}
=
u_t^{\text{target}}(x\mid z).
$$

整理一下就是

$$
u_t^{\text{target}}(x\mid z)
=
\left(
\dot\alpha_t-\frac{\dot\beta_t}{\beta_t}\alpha_t
\right)z
+
\frac{\dot\beta_t}{\beta_t}x,
$$

这正是式 (20)。

这就证明了它确实是条件高斯向量场。

### 对边缘向量场的直觉

下面来理解定理 9 的含义。

统计学中的 Bayes 公式告诉我们，下面这个量描述的是一个后验分布：

$$
\frac{p_t(x\mid z)p_{\text{data}}(z)}{p_t(x)}
$$

它可以理解为：

> “在观察到噪声化样本 $x$ 之后，这个 $x$ 来自数据点 $z$ 的后验概率”

其中 $p_{\text{data}}(z)$ 是先验分布。

因此，边缘向量场其实就是一个加权平均：

- 对每一个可能的数据点 $z$，我们都考虑一个速度 $u_t(x\mid z)$，也就是“把当前点 $x$ 朝着 $z$ 推过去”的方向；
- 然后用“我们相信 $x$ 来自 $z$ 的程度”去给这个方向加权；
- 最后对所有数据点 $z$ 求平均，就得到边缘向量场。

讲义接下来会把这个直觉严格化，并用连续性方程（continuity equation）来证明定理 9。

先定义散度算子（divergence）：

$$
\mathrm{div}(v_t)(x)
=
\sum_{i=1}^d \frac{\partial}{\partial x_i}v_t^i(x),
\tag{22}
$$

其中 $v_t^i$ 表示向量场 $v_t$ 的第 $i$ 个坐标分量。

### 定理 11（连续性方程）

考虑一个 flow model，其向量场为 $u_t^{\text{target}}$，并设

$$
X_0\sim p_{\text{init}}=p_0.
$$

则对所有 $0\le t\le 1$，有 $X_t\sim p_t$ 当且仅当满足

$$
\partial_t p_t(x)
=
-\mathrm{div}(p_t u_t^{\text{target}})(x),
\qquad \forall x\in\mathbb{R}^d,\ 0\le t\le 1.
\tag{23}
$$

这个式子就叫做连续性方程（continuity equation）。

#### 连续性方程的直觉

左边 $\partial_t p_t(x)$ 描述了点 $x$ 处的概率密度随时间变化的速度。

直觉上，这个变化应该等于“概率质量净流入的多少”。在 flow model 中，粒子 $X_t$ 会沿着向量场 $u_t^{\text{target}}$ 移动。物理里你可能记得：散度刻画的是一个向量场的净流出量，所以负散度就对应净流入量。

再乘上当前位置已有的概率质量 $p_t(x)$，就得到总的概率质量流入。由于概率质量守恒（总积分始终为 1），所以左右两边应该相等。

### 定理 9 的证明

根据定理 11，我们只需要证明按式 (18) 定义的边缘向量场满足连续性方程即可：

$$
\partial_t p_t(x)
=
\partial_t \int p_t(x\mid z)p_{\text{data}}(z)\,dz
=
\int \partial_t p_t(x\mid z)p_{\text{data}}(z)\,dz
$$

$$
=
-\int \mathrm{div}\bigl(p_t(\cdot\mid z)u_t^{\text{target}}(\cdot\mid z)\bigr)(x)\,p_{\text{data}}(z)\,dz
$$

$$
=
-\mathrm{div}\left(
\int p_t(x\mid z)u_t^{\text{target}}(x\mid z)p_{\text{data}}(z)\,dz
\right)
$$

$$
=
-\mathrm{div}\left(
p_t(x)\int u_t^{\text{target}}(x\mid z)\frac{p_t(x\mid z)p_{\text{data}}(z)}{p_t(x)}\,dz
\right)
$$

$$
=
-\mathrm{div}\bigl(p_t u_t^{\text{target}}\bigr)(x).
$$

于是连续性方程成立，由定理 11 可得定理 9。

## 3.3 学习边缘向量场

现在我们终于可以描述训练算法了。

Flow matching 的目标，是训练神经网络 $u_t^\theta$ 使它等于边缘向量场 $u_t^{\text{target}}$。如果这一点成立，那么根据定理 9，我们就知道终点满足

$$
X_1\sim p_{\text{data}}.
$$

下面记

$$
\mathrm{Unif}=\mathrm{Unif}[0,1]
$$

表示区间 $[0,1]$ 上的均匀分布，$E$ 表示随机变量的期望。

一种非常自然的做法，是用均方误差来逼近 $u_t^{\text{target}}$，也就是定义 flow matching loss：

$$
L_{\mathrm{FM}}(\theta)
=
\mathbb{E}_{t\sim \mathrm{Unif},\,x\sim p_t}
\bigl[
\|u_t^\theta(x)-u_t^{\text{target}}(x)\|^2
\bigr].
\tag{24}
$$

根据边缘路径的采样方式，也可以写成

$$
L_{\mathrm{FM}}(\theta)
=
\mathbb{E}_{t\sim \mathrm{Unif},\,z\sim p_{\text{data}},\,x\sim p_t(\cdot\mid z)}
\bigl[
\|u_t^\theta(x)-u_t^{\text{target}}(x)\|^2
\bigr].
\tag{25}
$$

直观地说，这个损失的含义是：

1. 先随机采一个时间 $t\in[0,1]$
2. 再从数据集里随机采一个数据点 $z$
3. 从 $p_t(\cdot\mid z)$ 中采样一个点 $x$（例如通过加噪）
4. 计算网络输出 $u_t^\theta(x)$
5. 最后把它和边缘向量场 $u_t^{\text{target}}(x)$ 做均方误差

但问题来了：虽然定理 9 给出了 $u_t^{\text{target}}(x)$ 的公式，我们却没法高效计算它，因为其中的积分是不可 tractable 的。

不过，条件向量场 $u_t^{\text{target}}(x\mid z)$ 通常是可解析计算的。于是，我们定义条件 flow matching loss：

$$
L_{\mathrm{CFM}}(\theta)
=
\mathbb{E}_{t\sim \mathrm{Unif},\,z\sim p_{\text{data}},\,x\sim p_t(\cdot\mid z)}
\bigl[
\|u_t^\theta(x)-u_t^{\text{target}}(x\mid z)\|^2
\bigr].
\tag{26}
$$

注意它和式 (24) 的区别：

- 式 (24) 用的是边缘向量场 $u_t^{\text{target}}(x)$
- 式 (26) 用的是条件向量场 $u_t^{\text{target}}(x\mid z)$

由于条件向量场有解析公式，所以式 (26) 很容易优化。

但这又引出一个新的问题：

> 如果我们关心的是边缘向量场，为什么可以只回归条件向量场？

答案是：对条件向量场做回归，其实等价于对边缘向量场做回归。下面这个结果把这个直觉精确地表达了出来。

### 定理 12

边缘 flow matching 损失与条件 flow matching 损失只差一个常数。也就是说，

$$
L_{\mathrm{FM}}(\theta)=L_{\mathrm{CFM}}(\theta)+C,
$$

其中 $C$ 与 $\theta$ 无关。

因此，它们的梯度相同：

$$
\nabla_\theta L_{\mathrm{FM}}(\theta)
=
\nabla_\theta L_{\mathrm{CFM}}(\theta).
$$

所以，用随机梯度下降等方法最小化 $L_{\mathrm{CFM}}(\theta)$，等价于最小化 $L_{\mathrm{FM}}(\theta)$。特别地，对于 $L_{\mathrm{CFM}}(\theta)$ 的最优解 $\theta^\*$，在模型表达能力无限强的理想情况下，我们会得到

$$
u_t^{\theta^\*}=u_t^{\text{target}},
$$

也就是神经网络学到了真正的边缘向量场。

### 直接证明

证明思路是：把均方误差展开成三项，然后把其中与 $\theta$ 无关的部分吸收到常数里。

首先，

$$
L_{\mathrm{FM}}(\theta)
=
\mathbb{E}_{t\sim \mathrm{Unif},\,x\sim p_t}
\bigl[
\|u_t^\theta(x)-u_t^{\text{target}}(x)\|^2
\bigr].
$$

展开平方项：

$$
L_{\mathrm{FM}}(\theta)
=
\mathbb{E}\bigl[\|u_t^\theta(x)\|^2
-2u_t^\theta(x)^\top u_t^{\text{target}}(x)
+\|u_t^{\text{target}}(x)\|^2\bigr].
$$

把最后一项单独记为常数 $C_1$，得到

$$
L_{\mathrm{FM}}(\theta)
=
\mathbb{E}\bigl[\|u_t^\theta(x)\|^2\bigr]
-2\mathbb{E}\bigl[u_t^\theta(x)^\top u_t^{\text{target}}(x)\bigr]
+C_1.
$$

第一项可以直接用边缘路径采样方式改写成

$$
\mathbb{E}_{t,z,x\sim p_t(\cdot\mid z)}\bigl[\|u_t^\theta(x)\|^2\bigr].
$$

关键在第二项。先把它写成积分：

$$
\mathbb{E}_{t,x\sim p_t}\bigl[u_t^\theta(x)^\top u_t^{\text{target}}(x)\bigr]
=
\int_0^1 \int p_t(x)\,u_t^\theta(x)^\top u_t^{\text{target}}(x)\,dx\,dt.
$$

再代入边缘向量场的定义式 (18)：

$$
=
\int_0^1\int
p_t(x)\,u_t^\theta(x)^\top
\left[
\int
u_t^{\text{target}}(x\mid z)\frac{p_t(x\mid z)p_{\text{data}}(z)}{p_t(x)}\,dz
\right]
dx\,dt.
$$

交换积分次序后，可得

$$
=
\mathbb{E}_{t\sim \mathrm{Unif},\,z\sim p_{\text{data}},\,x\sim p_t(\cdot\mid z)}
\bigl[
u_t^\theta(x)^\top u_t^{\text{target}}(x\mid z)
\bigr].
$$

把这个结果代回去，就得到

$$
L_{\mathrm{FM}}(\theta)
=
\mathbb{E}_{t,z,x\sim p_t(\cdot\mid z)}\bigl[\|u_t^\theta(x)\|^2\bigr]
-2\mathbb{E}_{t,z,x\sim p_t(\cdot\mid z)}
\bigl[u_t^\theta(x)^\top u_t^{\text{target}}(x\mid z)\bigr]
+C_1.
$$

接着我们加上再减去

$$
\|u_t^{\text{target}}(x\mid z)\|^2
$$

这个同一个量，就能重新拼成一个平方项：

$$
L_{\mathrm{FM}}(\theta)
=
\mathbb{E}_{t,z,x\sim p_t(\cdot\mid z)}
\bigl[
\|u_t^\theta(x)-u_t^{\text{target}}(x\mid z)\|^2
\bigr]
+C_2+C_1.
$$

于是

$$
L_{\mathrm{FM}}(\theta)
=
L_{\mathrm{CFM}}(\theta)+C,
$$

其中 $C=C_1+C_2$ 与 $\theta$ 无关。

证毕。

因此，flow matching 的训练过程，就是最小化条件 flow matching loss。

这个算法有几个非常显著的优点：

1. 训练时完全不需要真的去模拟 ODE。人们把这个特点称为 simulation-free。
2. 训练目标只是一个简单回归问题，本质上就是回归到 $u_t^{\text{target}}(x\mid z)$。
3. 整个训练流程极其简单。

这些特点使 flow matching 成为训练大规模生成模型时非常有吸引力的方法。

在训练完 $u_t^\theta$ 之后，我们就可以通过模拟 flow model

$$
\frac{d}{dt}X_t = u_t^\theta(X_t),\qquad
X_0\sim p_{\text{init}}
\tag{27}
$$

来得到样本 $X_1\sim p_{\text{data}}$。这整套流程在文献中就叫做 flow matching。

### 算法 3：Flow Matching 训练流程（针对 Gaussian CondOT 路径 $p_t(x\mid z)=\mathcal{N}(tz,(1-t)^2)$）

输入：样本数据集 $z\sim p_{\text{data}}$，神经网络 $u_t^\theta$

1. 对每个 mini-batch：
2. 从数据集中采样一个数据样本 $z$
3. 采样一个随机时间

$$
t\sim \mathrm{Unif}[0,1]
$$

4. 采样噪声

$$
\epsilon\sim \mathcal{N}(0,I_d)
$$

5. 构造

$$
x=tz+(1-t)\epsilon
\qquad
\text{（一般情形下： }x\sim p_t(\cdot\mid z)\text{）}
$$

6. 计算损失

$$
L(\theta)
=
\|u_t^\theta(x)-(z-\epsilon)\|^2
\qquad
\text{（一般情形下： }\|u_t^\theta(x)-u_t^{\text{target}}(x\mid z)\|^2\text{）}
$$

7. 用梯度更新参数 $\theta$
8. 重复

### 例 13（高斯条件概率路径下的 Flow Matching）

回到高斯概率路径

$$
p_t(\cdot\mid z)=\mathcal{N}(\alpha_t z,\beta_t^2 I_d).
$$

从条件路径采样，可以写成

$$
\epsilon\sim \mathcal{N}(0,I_d)
\quad\Longrightarrow\quad
x_t=\alpha_t z+\beta_t\epsilon
\sim
\mathcal{N}(\alpha_t z,\beta_t^2 I_d)
=
p_t(\cdot\mid z).
\tag{28}
$$

而我们在式 (20) 中已经推导出条件向量场为

$$
u_t^{\text{target}}(x\mid z)
=
\left(
\dot\alpha_t-\frac{\dot\beta_t}{\beta_t}\alpha_t
\right)z
+
\frac{\dot\beta_t}{\beta_t}x.
\tag{29}
$$

把这个代入条件 flow matching loss，就得到

$$
L_{\mathrm{CFM}}(\theta)
=
\mathbb{E}_{t\sim \mathrm{Unif},\,z\sim p_{\text{data}},\,x\sim \mathcal{N}(\alpha_t z,\beta_t^2 I_d)}
\left[
\left\|
u_t^\theta(x)
-
\left(
\dot\alpha_t-\frac{\dot\beta_t}{\beta_t}\alpha_t
\right)z
-
\frac{\dot\beta_t}{\beta_t}x
\right\|^2
\right].
\tag{30}
$$

再把 $x=\alpha_t z+\beta_t\epsilon$ 代入，就得到

$$
L_{\mathrm{CFM}}(\theta)
=
\mathbb{E}_{t\sim \mathrm{Unif},\,z\sim p_{\text{data}},\,\epsilon\sim \mathcal{N}(0,I_d)}
\left[
\left\|
u_t^\theta(\alpha_t z+\beta_t\epsilon)
-
\bigl(\dot\alpha_t z+\dot\beta_t\epsilon\bigr)
\right\|^2
\right].
\tag{31}
$$

可以看到，这个损失非常简单：

- 采一个数据点 $z$
- 采一些噪声 $\epsilon$
- 然后做一个均方误差

再看一个特别常见的特例：令

$$
\alpha_t=t,\qquad \beta_t=1-t.
$$

对应的概率路径

$$
p_t(x\mid z)=\mathcal{N}(tz,(1-t)^2)
$$

有时被称为 Gaussian CondOT probability path。

这时有

$$
\dot\alpha_t=1,\qquad \dot\beta_t=-1,
$$

于是损失进一步简化为

$$
L_{\mathrm{CFM}}(\theta)
=
\mathbb{E}_{t\sim \mathrm{Unif},\,z\sim p_{\text{data}},\,\epsilon\sim \mathcal{N}(0,I_d)}
\left[
\|u_t^\theta(tz+(1-t)\epsilon)-(z-\epsilon)\|^2
\right].
$$

很多著名的最先进模型都使用了这个简单但高效的训练过程，例如 Stable Diffusion 3、Meta 的 Movie Gen Video，以及很多可能尚未公开的专有模型。

## 小结 14（Flow Matching）

Flow matching 的训练目标，是学习边缘向量场 $u_t^{\text{target}}$。

为此，我们先选择一条条件概率路径 $p_t(x\mid z)$，使其满足

$$
p_0(\cdot\mid z)=p_{\text{init}},\qquad
p_1(\cdot\mid z)=\delta_z.
$$

接着，找到一个条件向量场 $u_t^{\text{target}}(x\mid z)$，使得它对应的 flow $\psi_t^{\text{target}}(x\mid z)$ 满足

$$
X_0\sim p_{\text{init}}
\quad\Longrightarrow\quad
X_t=\psi_t^{\text{target}}(X_0\mid z)\sim p_t(\cdot\mid z).
$$

等价地，也可以说：$u_t^{\text{target}}$ 满足连续性方程。

然后定义边缘向量场

$$
u_t^{\text{target}}(x)
=
\int
u_t^{\text{target}}(x\mid z)
\frac{p_t(x\mid z)p_{\text{data}}(z)}{p_t(x)}\,dz.
\tag{32}
$$

这个边缘向量场会沿着边缘概率路径演化，也就是

$$
X_0\sim p_{\text{init}},\qquad
dX_t = u_t^{\text{target}}(X_t)\,dt
\quad\Longrightarrow\quad
X_t\sim p_t,\qquad 0\le t\le 1.
\tag{33}
$$

特别地，

$$
X_1\sim p_{\text{data}},
$$

所以它确实实现了“把噪声变成数据”。

为了学习它，我们最小化条件 flow matching loss：

$$
L_{\mathrm{CFM}}(\theta)
=
\mathbb{E}_{t\sim \mathrm{Unif},\,z\sim p_{\text{data}},\,x\sim p_t(\cdot\mid z)}
\bigl[
\|u_t^\theta(x)-u_t^{\text{target}}(x\mid z)\|^2
\bigr].
\tag{34}
$$

最常用的例子是高斯概率路径。在这种情况下，公式变为：

$$
p_t(x\mid z)=\mathcal{N}(x;\alpha_t z,\beta_t^2 I_d)
\tag{35}
$$

$$
u_t^{\text{flow}}(x\mid z)
=
\left(
\dot\alpha_t-\frac{\dot\beta_t}{\beta_t}\alpha_t
\right)z
+
\frac{\dot\beta_t}{\beta_t}x
\tag{36}
$$

$$
L_{\mathrm{CFM}}(\theta)
=
\mathbb{E}_{t\sim \mathrm{Unif},\,z\sim p_{\text{data}},\,\epsilon\sim \mathcal{N}(0,I_d)}
\left[
\left\|
u_t^\theta(\alpha_t z+\beta_t\epsilon)
-
\bigl(\dot\alpha_t z+\dot\beta_t\epsilon\bigr)
\right\|^2
\right]
\tag{37}
$$

其中 $\alpha_t,\beta_t\in\mathbb{R}$ 是我们自己选择的噪声调度函数，它们连续可微、单调，并满足

$$
\alpha_0=\beta_1=0,\qquad
\alpha_1=\beta_0=1.
$$

例如，一个最常见的选择就是

$$
\alpha_t=t,\qquad \beta_t=1-t.
$$
