# 第 2 节 Flow 与 Diffusion 模型（中文翻译）

对应原讲义 `lecture_notes.pdf` 第 7-13 页，即 Section 2：`Flow and Diffusion Models`，包含 2.1、2.2 两个小节。

## 2 Flow 与 Diffusion 模型

在上一节中，我们把生成建模形式化为：从数据分布 $p_{\text{data}}$ 中采样。我们也明确了目标：构造一个生成模型，也就是一个能够返回样本

$$
z\sim p_{\text{data}}
$$

的算法。

这一节中，我们将说明：如何把一个生成模型构造成某个精心设计的微分方程的数值模拟过程。举例来说，flow matching 和 diffusion model 分别对应于模拟常微分方程（ODE）和随机微分方程（SDE）。

因此，本节的目标是定义并构造这些生成模型，因为它们会贯穿后续整份讲义。更具体地说：

1. 我们首先定义 ODE 和 SDE，并讨论如何模拟它们。
2. 接着，我们说明如何用深度神经网络对 ODE/SDE 进行参数化。
3. 由此得到 flow model 和 diffusion model 的定义，以及从这些模型中采样的基本算法。

在后面的章节中，我们再进一步讨论如何训练这些模型。

## 2.1 Flow 模型

我们先从常微分方程（ODE）开始。

一个 ODE 的解由一条轨迹（trajectory）给出，也就是形如

$$
X:[0,1]\to \mathbb{R}^d,\qquad t\mapsto X_t
$$

的函数。它把时间 $t$ 映射为欧氏空间 $\mathbb{R}^d$ 中的一个位置。

每个 ODE 都由一个向量场（vector field）$u$ 定义。它的形式是

$$
u:\mathbb{R}^d\times [0,1]\to \mathbb{R}^d,\qquad
(x,t)\mapsto u_t(x),
$$

也就是说：对任意时刻 $t$ 和任意位置 $x$，我们都会得到一个向量

$$
u_t(x)\in\mathbb{R}^d,
$$

它指定了空间中的一个速度方向。

ODE 对轨迹施加了一个约束：我们希望这条轨迹从起点 $x_0$ 出发，并且始终“沿着”向量场 $u_t$ 指定的方向运动。形式化地说，这样的轨迹是如下方程的解：

$$
\frac{d}{dt}X_t = u_t(X_t)
\qquad\triangleright\ \text{ODE}
\tag{1a}
$$

$$
X_0 = x_0
\qquad\triangleright\ \text{初始条件}
\tag{1b}
$$

式 (1a) 要求 $X_t$ 的导数由向量场 $u_t$ 给出的方向决定；式 (1b) 要求在 $t=0$ 时从 $x_0$ 出发。

现在我们会问：如果在 $t=0$ 时从 $X_0=x_0$ 出发，那么在时刻 $t$ 我们会到哪里，即 $X_t$ 是什么？这个问题由一个叫做流（flow）的函数来回答，它是如下 ODE 的解：

$$
\psi:\mathbb{R}^d\times [0,1]\to \mathbb{R}^d,\qquad
(x_0,t)\mapsto \psi_t(x_0)
\tag{2a}
$$

$$
\frac{d}{dt}\psi_t(x_0)=u_t(\psi_t(x_0))
\qquad\triangleright\ \text{流的 ODE}
\tag{2b}
$$

$$
\psi_0(x_0)=x_0
\qquad\triangleright\ \text{流的初始条件}
\tag{2c}
$$

对于给定的初始条件 $X_0=x_0$，ODE 的轨迹可写为

$$
X_t=\psi_t(X_0).
$$

因此，从直观上看，向量场、ODE 和流，其实是同一个对象的三种描述方式：

- 向量场定义了 ODE；
- ODE 的解是流。

和研究任何方程一样，我们自然会问：解是否存在？如果存在，是否唯一？一个基础数学结论告诉我们，在很弱的条件下，答案都是肯定的。

### 定理 3（流的存在性与唯一性）

如果

$$
u:\mathbb{R}^d\times [0,1]\to \mathbb{R}^d
$$

是连续可微的，并且其导数有界，那么式 (2b)-(2c) 中的流 ODE 有唯一解，这个解由一个流 $\psi_t$ 给出。并且此时 $\psi_t$ 对每个 $t$ 都是一个微分同胚（diffeomorphism），也就是说它连续可微，且其逆映射 $\psi_t^{-1}$ 也连续可微。

注意，在机器学习中，这些存在唯一性所需的假设几乎总是满足的，因为我们通常用神经网络来参数化 $u_t(x)$，而它们通常具有有界导数。因此，定理 3 对你来说更多是个好消息，而不是负担：在我们关心的场景里，flow 存在，而且是 ODE 的唯一解。证明可见 [32, 9]。

### 例 4（线性向量场）

考虑一个简单的向量场

$$
u_t(x)=-\theta x,\qquad \theta>0.
$$

它关于 $x$ 是线性的。此时函数

$$
\psi_t(x_0)=\exp(-\theta t)x_0
\tag{3}
$$

定义了一个满足式 (2b)-(2c) 的流。

你可以自己检查：

1. 当 $t=0$ 时，

$$
\psi_0(x_0)=x_0
$$

2. 对时间求导：

$$
\frac{d}{dt}\psi_t(x_0)
=
\frac{d}{dt}\bigl(\exp(-\theta t)x_0\bigr)
=
-\theta \exp(-\theta t)x_0
=
-\theta \psi_t(x_0)
=
u_t(\psi_t(x_0)).
$$

这里用到了链式法则。

图 3 展示了这种流的形状：它会以指数速度收敛到原点 0。

### ODE 的模拟

一般来说，当 $u_t$ 不像上面的例子那样简单时，我们通常无法显式写出流 $\psi_t$。这时就需要用数值方法来模拟 ODE。幸运的是，这在数值分析中是一个非常经典且研究充分的话题，已经有大量强大的方法 [21]。

其中最简单也最直观的方法之一是 Euler 方法。它的做法是：初始化

$$
X_0=x_0
$$

然后按下面的规则迭代更新：

$$
X_{t+h}=X_t + h\,u_t(X_t),
\qquad
t=0,h,2h,3h,\ldots,1-h
\tag{4}
$$

其中

$$
h=n^{-1}>0
$$

是步长，$n\in\mathbb{N}$ 是模拟步数。

在这门课里，Euler 方法已经足够用了。

为了让你感受一种稍复杂一点的方法，我们再看 Heun 方法，其更新规则为：

$$
X'_{t+h}=X_t + h\,u_t(X_t)
\qquad\triangleright\ \text{对新状态的初步猜测（与 Euler 步相同）}
$$

$$
X_{t+h}=X_t+\frac{h}{2}\Bigl(u_t(X_t)+u_{t+h}(X'_{t+h})\Bigr)
\qquad\triangleright\ \text{用当前位置与猜测位置上的平均向量场更新}
$$

直观地说，Heun 方法会先做一个粗略预测 $X'_{t+h}$，再利用这个预测点的信息来修正最开始的方向。

### Flow 模型

现在，我们可以利用 ODE 来构造一个生成模型：只需把向量场换成一个神经网络向量场 $u_t^\theta$。

这里暂时只需把它理解为一个带参数 $\theta$ 的函数：

$$
u_t^\theta:\mathbb{R}^d\times [0,1]\to \mathbb{R}^d.
$$

后面我们会再讨论具体用什么神经网络架构。

回忆一下我们的目标：生成来自数据分布 $p_{\text{data}}$ 的样本

$$
z\sim p_{\text{data}}.
$$

特别地，这些样本必须是随机的。但 ODE 本身是完全确定性的，并不包含随机性。为了引入随机性，我们只需要把初始条件设为随机变量。

具体地，我们选择一个初始分布 $p_{\text{init}}$。在多数情况下，我们会取

$$
p_{\text{init}}=\mathcal{N}(0,I_d),
$$

即标准高斯分布。最重要的是，无论你选什么初始分布，它都必须是在推理阶段容易采样的。

于是，一个 flow model 可以写成：

$$
X_0\sim p_{\text{init}}
\qquad\triangleright\ \text{随机初始化}
$$

$$
\frac{d}{dt}X_t = u_t^\theta(X_t)
\qquad\triangleright\ \text{ODE}
$$

我们的目标是让轨迹终点 $X_1$ 的分布变成 $p_{\text{data}}$，也就是

$$
X_1\sim p_{\text{data}}
\qquad\Longleftrightarrow\qquad
\psi_1^\theta(X_0)\sim p_{\text{data}},
$$

其中 $\psi_t^\theta$ 表示由 $u_t^\theta$ 诱导出的流。

不过注意：虽然它叫做 flow model，但神经网络参数化的其实是向量场，而不是流本身。要得到流，我们必须对 ODE 进行数值模拟。

### 算法 1：用 Euler 方法从 Flow 模型中采样

输入：神经网络向量场 $u_t^\theta$、步数 $n$

1. 令 $t=0$
2. 令步长

$$
h=\frac{1}{n}
$$

3. 采样初值

$$
X_0\sim p_{\text{init}}
$$

4. 对 $i=1,\ldots,n$：

$$
X_{t+h}=X_t + h\,u_t^\theta(X_t)
$$

并更新

$$
t\leftarrow t+h
$$

5. 返回 $X_1$

## 2.2 Diffusion 模型

随机微分方程（SDE）可以看作是在 ODE 的确定性轨迹基础上，引入随机性的扩展。SDE 的轨迹通常称为随机过程（stochastic process）$(X_t)_{0\le t\le 1}$，满足：

- 对每个 $0\le t\le 1$，$X_t$ 都是一个随机变量；
- 对每次随机采样而言，映射

$$
X:[0,1]\to \mathbb{R}^d,\qquad t\mapsto X_t
$$

是一条随机轨迹。

特别地，如果我们对同一个随机过程模拟两次，得到的结果可能不同，因为它的动力学本身就是随机的。

### 布朗运动

SDE 是通过布朗运动（Brownian motion）构造出来的。布朗运动是一个非常基础的随机过程，最早来源于对物理扩散过程的研究。你可以把布朗运动想象成“连续版的随机游走”。

形式上，一个布朗运动

$$
W=(W_t)_{0\le t\le 1}
$$

是一个随机过程，满足：

- $W_0=0$
- 轨迹 $t\mapsto W_t$ 是连续的
- 以及以下两个条件：

1. 正态增量（normal increments）：

$$
W_t-W_s \sim \mathcal{N}(0,(t-s)I_d),
\qquad \forall\ 0\le s<t
$$

也就是说，增量服从高斯分布，其方差随时间线性增长。

2. 独立增量（independent increments）：

对任意

$$
0\le t_0<t_1<\cdots<t_n=1,
$$

增量

$$
W_{t_1}-W_{t_0},\ldots,W_{t_n}-W_{t_{n-1}}
$$

是彼此独立的随机变量。

布朗运动也叫 Wiener 过程，所以我们用字母 `W` 来表示它。讲义里还补充了一句：Norbert Wiener 是 MIT 一位著名数学家，在 MIT 数学系还能看到他的画像。

我们可以用一个简单的离散近似来模拟布朗运动。设步长 $h>0$，令

$$
W_0=0
$$

并更新

$$
W_{t+h}=W_t+\sqrt{h}\,\epsilon_t,
\qquad
\epsilon_t\sim \mathcal{N}(0,I_d),
\qquad
t=0,h,2h,\ldots,1-h.
\tag{5}
$$

图 2 展示了一些一维布朗运动轨迹的样子。

布朗运动在随机过程中的地位，就像高斯分布在概率论中的地位一样核心。它在机器学习之外也有非常广泛的应用，比如金融、统计物理和流行病学。在金融里，布朗运动常被用来描述复杂金融工具的价格变化。

从数学角度看，布朗运动本身也很迷人。比如，它的路径虽然是连续的，也就是说你可以不抬笔一直画下去；但它们的长度却是无穷的，也就是说你又永远画不完。

### 从 ODE 到 SDE

SDE 的核心思想，是在 ODE 的确定性动力学上加上一部分由布朗运动驱动的随机动力学。

因为一切都变成随机的了，我们不能再像式 (1a) 那样直接写导数形式。因此，我们先把 ODE 的轨迹 $(X_t)_{0\le t\le 1}$ 改写成一个不显式使用导数的形式：

$$
\frac{d}{dt}X_t = u_t(X_t)
\quad\Longleftrightarrow\quad
\frac{1}{h}(X_{t+h}-X_t)=u_t(X_t)+R_t(h)
$$

进一步等价于

$$
X_{t+h}=X_t + h\,u_t(X_t)+hR_t(h),
$$

其中 $R_t(h)$ 表示一个在 $h$ 很小时可以忽略的函数，也就是

$$
\lim_{h\to 0}R_t(h)=0.
$$

这其实只是把我们已经知道的东西重新说了一遍：ODE 的轨迹在每一个小时间步里，都会沿着 $u_t(X_t)$ 的方向走一小步。

现在，我们可以把最后一个式子改造成随机版本：SDE 的轨迹在每一步中，除了沿 $u_t(X_t)$ 的方向前进一步，还会再叠加一部分由布朗运动带来的随机变化：

$$
X_{t+h}
=
X_t
+ h\,u_t(X_t)
+ \sigma_t(W_{t+h}-W_t)
+ hR_t(h),
\tag{6}
$$

其中：

- $h\,u_t(X_t)$ 是确定性部分；
- $\sigma_t(W_{t+h}-W_t)$ 是随机部分；
- $\sigma_t\ge 0$ 称为扩散系数（diffusion coefficient）；
- $R_t(h)$ 是随机误差项，满足其标准差

$$
\mathbb{E}\bigl[\|R_t(h)\|^2\bigr]^{1/2}\to 0
\qquad \text{当 } h\to 0.
$$

上面的式子就描述了一个随机微分方程（SDE）。我们通常把它用如下符号化写法表示：

$$
dX_t = u_t(X_t)\,dt + \sigma_t\,dW_t
\qquad\triangleright\ \text{SDE}
\tag{7a}
$$

$$
X_0=x_0
\qquad\triangleright\ \text{初始条件}
\tag{7b}
$$

不过一定要记住，上面的 $dX_t$ 记号只是式 (6) 的一种非正式、符号化写法。

不幸的是，SDE 不再拥有像 ODE 那样的流映射 $\phi_t$。原因在于：即使知道了 $X_0\sim p_{\text{init}}$，$X_t$ 的值也不再被完全决定，因为整个演化过程本身就是随机的。

不过，就像 ODE 一样，我们仍有一个存在性与唯一性的结论。

### 定理 5（SDE 解的存在性与唯一性）

如果

$$
u:\mathbb{R}^d\times [0,1]\to \mathbb{R}^d
$$

连续可微且导数有界，并且 $\sigma_t$ 是连续函数，那么式 (7a)-(7b) 所示的 SDE 存在唯一解，即存在唯一一个随机过程 $(X_t)_{0\le t\le 1}$ 满足式 (6)。

如果这是一门随机微积分课，我们可能要花好几讲来证明这个定理，并从第一性原理严格构造布朗运动和随机积分，从而严格构造过程 $X_t$。但由于本课程重点在机器学习，我们把更技术化的处理留给参考文献 [29]。

最后还要注意：每个 ODE 其实也可以看作一个 SDE，只要它的扩散系数为

$$
\sigma_t=0.
$$

因此在后续课程中，当我们说 SDE 时，也会把 ODE 视为它的一个特例。

### 例 6（Ornstein-Uhlenbeck 过程）

考虑常数扩散系数

$$
\sigma_t=\sigma\ge 0
$$

以及线性漂移（drift）

$$
u_t(x)=-\theta x,\qquad \theta>0.
$$

这时对应的 SDE 为

$$
dX_t = -\theta X_t\,dt + \sigma\,dW_t.
\tag{8}
$$

它的解 $(X_t)_{0\le t\le 1}$ 被称为 Ornstein-Uhlenbeck（OU）过程。图 3 对它进行了可视化。

这个过程的直观含义是：

- 向量场 $-\theta x$ 会把过程拉回中心 0，因为漂移总是指向当前位置的反方向；
- 扩散系数 $\sigma$ 则会不断往系统里加入新的噪声。

当我们把这个过程模拟到 $t\to\infty$ 时，它会收敛到一个高斯分布

$$
\mathcal{N}\Bigl(0,\frac{\sigma^2}{2\theta}\Bigr).
$$

注意，当 $\sigma=0$ 时，这就退化为一个 flow，也就是我们在式 (3) 中看到过的线性向量场情形。

### SDE 的模拟

如果你觉得到目前为止 SDE 的定义还有点抽象，也完全不用担心。理解 SDE 的一个更直观方式，是直接问：我们应该如何模拟它？

最简单的方法叫做 Euler-Maruyama 方法。它对 SDE 的地位，就像 Euler 方法对 ODE 的地位一样。

使用 Euler-Maruyama 方法时，我们从

$$
X_0=x_0
$$

开始，然后迭代更新：

$$
X_{t+h}=X_t + h\,u_t(X_t) + \sqrt{h}\,\sigma_t\,\epsilon_t,
\qquad
\epsilon_t\sim \mathcal{N}(0,I_d),
\tag{9}
$$

其中

$$
h=n^{-1}>0
$$

是步长超参数，$n\in\mathbb{N}$。

换句话说，Euler-Maruyama 的每一步会做两件事：

1. 沿着 $u_t(X_t)$ 的方向走一小步；
2. 再加上一点按 $\sqrt{h}\sigma_t$ 缩放的高斯噪声。

在这门课里，当我们需要模拟 SDE 时（例如在配套实验中），通常都会使用 Euler-Maruyama 方法。

### Diffusion 模型

现在，我们可以像构造 ODE 生成模型那样，利用 SDE 来构造一个生成模型。

回忆我们的目标：把一个简单分布 $p_{\text{init}}$ 变换成一个复杂分布 $p_{\text{data}}$。和 ODE 一样，从一个随机初始化

$$
X_0\sim p_{\text{init}}
$$

开始模拟 SDE，是实现这种变换的自然方法。

为了参数化这个 SDE，我们只需要参数化其核心成分之一，也就是向量场 $u_t$。于是我们用神经网络 $u_t^\theta$ 来替代它。

因此，一个 diffusion model 定义为：

$$
X_0\sim p_{\text{init}}
\qquad\triangleright\ \text{随机初始化}
$$

$$
dX_t = u_t^\theta(X_t)\,dt+\sigma_t\,dW_t
\qquad\triangleright\ \text{SDE}
$$

算法 2 给出了如何用 Euler-Maruyama 方法从 diffusion model 采样。

### 算法 2：用 Euler-Maruyama 方法从 Diffusion 模型中采样

输入：神经网络 $u_t^\theta$、步数 $n$、扩散系数 $\sigma_t$

1. 令 $t=0$
2. 令步长

$$
h=\frac{1}{n}
$$

3. 采样

$$
X_0\sim p_{\text{init}}
$$

4. 对 $i=1,\ldots,n$：

5. 采样

$$
\epsilon\sim \mathcal{N}(0,I_d)
$$

6. 更新

$$
X_{t+h}=X_t + h\,u_t^\theta(X_t)+\sigma_t\sqrt{h}\,\epsilon
$$

7. 更新

$$
t\leftarrow t+h
$$

8. 返回 $X_1$

## 小结 7（SDE 生成模型）

在整份讲义中，一个 diffusion model 由两部分组成：

1. 一个带参数 $\theta$ 的神经网络 $u^\theta_t$，它参数化了向量场

$$
u^\theta:\mathbb{R}^d\times [0,1]\to \mathbb{R}^d,\qquad
(x,t)\mapsto u_t^\theta(x)
$$

2. 一个固定的扩散系数 $\sigma_t$：

$$
\sigma_t:[0,1]\to [0,\infty),\qquad t\mapsto \sigma_t
$$

为了从这个 SDE 模型中得到样本（也就是生成对象），步骤如下：

- 初始化：

$$
X_0\sim p_{\text{init}}
$$

通常取简单分布，例如高斯分布。

- 模拟：

$$
dX_t = u_t^\theta(X_t)\,dt+\sigma_t\,dW_t
$$

从 $t=0$ 一直模拟到 $t=1$。

- 目标：

$$
X_1\sim p_{\text{data}}
$$

也就是说，我们希望终点分布变成数据分布。

最后，一个很重要的关系是：

### 当 $\sigma_t=0$ 时，diffusion model 就退化为 flow model。
