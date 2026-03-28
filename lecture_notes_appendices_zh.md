# 附录 A-E（中文翻译）

对应原讲义 `lecture_notes.pdf` 第 70-84 页，包含附录 A、B、C、D、E。

## 附录 A 概率论简要回顾

这一部分简要回顾一些基础概率论概念。本节部分内容改写自 [26]。

### A.1 随机向量

考虑 $d$ 维欧氏空间中的数据

$$
x=(x_1,\ldots,x_d)\in \mathbb{R}^d
$$

其标准欧氏内积与范数定义为

$$
\langle x,y\rangle=\sum_{i=1}^d x_i y_i,
\qquad
\|x\|=\sqrt{\langle x,x\rangle}.
$$

我们考虑具有连续概率密度函数（PDF）的随机变量（RV）

$$
X\in \mathbb{R}^d.
$$

它的密度函数记为

$$
p_X:\mathbb{R}^d\to \mathbb{R}_{\ge 0},
$$

并且对任意事件集合 $A$，有

$$
\mathbb{P}(X\in A)=\int_A p_X(x)\,dx,
\tag{96}
$$

同时满足

$$
\int p_X(x)\,dx=1.
$$

按惯例，当积分区域是整个空间 $\mathbb{R}^d$ 时，我们会省略积分区间。为了记号简洁，讲义中把随机变量 $X_t$ 的 PDF $p_{X_t}$ 简写成 $p_t$。

我们用

$$
X\sim p
\quad\text{或}\quad
X\sim p(X)
$$

表示 $X$ 服从分布 $p$。

生成建模中最常见的一类分布是 $d$ 维各向同性高斯：

$$
\mathcal{N}(x;\mu,\sigma^2 I)
=
(2\pi \sigma^2)^{-d/2}
\exp\left(
-\frac{\|x-\mu\|^2}{2\sigma^2}
\right),
\tag{97}
$$

其中 $\mu\in\mathbb{R}^d$ 是均值，$\sigma>0$ 是标准差。

随机变量的期望，是在最小二乘意义下最接近 $X$ 的常向量：

$$
\mathbb{E}[X]
=
\arg\min_{z\in\mathbb{R}^d}
\int \|x-z\|^2 p_X(x)\,dx
=
\int x\,p_X(x)\,dx.
\tag{98}
$$

计算随机变量函数期望时，一个非常常用的工具是“无意识统计学家法则”（LOTUS）：

$$
\mathbb{E}[f(X)]
=
\int f(x)p_X(x)\,dx.
\tag{99}
$$

必要时，我们也会显式写成 $\mathbb{E}_X[f(X)]$ 来强调期望是对哪个随机变量取的。

### A.2 条件密度与条件期望

给定两个随机变量

$$
X,Y\in\mathbb{R}^d,
$$

它们的联合密度 $p_{X,Y}(x,y)$ 的边缘分布满足

$$
\int p_{X,Y}(x,y)\,dy = p_X(x),
\qquad
\int p_{X,Y}(x,y)\,dx = p_Y(y).
\tag{100}
$$

当 $p_Y(y)>0$ 时，条件密度定义为

$$
p_{X\mid Y}(x\mid y)
:=
\frac{p_{X,Y}(x,y)}{p_Y(y)}.
\tag{101}
$$

类似地也可以定义 $p_{Y\mid X}$。

Bayes 公式给出

$$
p_{Y\mid X}(y\mid x)
=
\frac{p_{X\mid Y}(x\mid y)p_Y(y)}{p_X(x)},
\qquad p_X(x)>0.
\tag{102}
$$

#### 条件期望

条件期望 $\mathbb{E}[X\mid Y]$ 可以理解为：在最小二乘意义下，$X$ 用某个关于 $Y$ 的函数 $g(Y)$ 来近似时最优的选择。

形式上，

$$
g^\star
:=
\arg\min_{g:\mathbb{R}^d\to\mathbb{R}^d}
\mathbb{E}\|X-g(Y)\|^2.
\tag{103}
$$

于是对任意满足 $p_Y(y)>0$ 的 $y$，有

$$
\mathbb{E}[X\mid Y=y]
:=
g^\star(y)
=
\int x\,p_{X\mid Y}(x\mid y)\,dx.
\tag{104}
$$

把 $g^\star$ 与随机变量 $Y$ 复合起来，就得到

$$
\mathbb{E}[X\mid Y]
:=
g^\star(Y),
\tag{105}
$$

它本身是一个随机变量。

这里要特别注意：

- $\mathbb{E}[X\mid Y=y]$ 是一个函数值；
- $\mathbb{E}[X\mid Y]$ 是一个随机变量。

虽然在日常说法里两者都常被叫“条件期望”，但它们是不同对象。

#### 塔式性质

条件期望有一个非常好用的性质，叫做塔式性质（tower property）：

$$
\mathbb{E}[\mathbb{E}[X\mid Y]]
=
\mathbb{E}[X].
\tag{106}
$$

因为 $\mathbb{E}[X\mid Y]$ 本身也是随机变量，所以外层期望就是对它再取一次平均。

#### 一个常用恒等式

对任意随机变量函数 $f(X,Y)$，有

$$
\mathbb{E}[f(X,Y)\mid Y=y]
=
\int f(x,y)p_{X\mid Y}(x\mid y)\,dx.
\tag{107}
$$

这条公式在推导条件期望相关表达式时非常有用。

## 附录 B Fokker-Planck 方程的证明

这一节给出 Fokker-Planck 方程的一个自包含证明，其中连续性方程（定理 11）是它的特殊情形。

讲义特别提醒：这部分并不是理解正文所必需的，而且数学上更偏进阶。如果你想更深入理解 Fokker-Planck 方程从哪里来，这一节会很有帮助。

### 定理 41（Fokker-Planck 方程）

设 $p_t$ 是一条概率路径，且 $p_0=p_{\text{init}}$。考虑 SDE

$$
X_0\sim p_{\text{init}},\qquad
dX_t = u_t(X_t)\,dt+\sigma_t\,dW_t.
$$

那么，对所有 $0\le t\le 1$，$X_t$ 服从分布 $p_t$ 当且仅当 Fokker-Planck 方程成立：

$$
\partial_t p_t(x)
=
-\mathrm{div}(p_tu_t)(x)
+
\frac{\sigma_t^2}{2}\Delta p_t(x),
\qquad
\forall x\in\mathbb{R}^d,\ 0\le t\le 1.
\tag{108}
$$

### 证明思路

先证明“必要性”：如果 $X_t\sim p_t$，那么 Fokker-Planck 方程成立。

证明技巧是引入测试函数（test functions）$f$：也就是光滑且具有紧支撑的函数

$$
f:\mathbb{R}^d\to \mathbb{R}.
$$

它的作用是把“逐点相等”转化成“对任意测试函数积分后相等”：

$$
g_1(x)=g_2(x)\ \forall x
\Longleftrightarrow
\int f(x)g_1(x)\,dx
=
\int f(x)g_2(x)\,dx
\quad\text{对所有测试函数 }f.
\tag{109}
$$

有了测试函数之后，我们就可以安全地做积分分部。

对可积函数 $f_1,f_2$，有

$$
\int f_1(x)\frac{\partial}{\partial x_i}f_2(x)\,dx
=
-\int f_2(x)\frac{\partial}{\partial x_i}f_1(x)\,dx.
\tag{110}
$$

由此可得两个常用恒等式：

$$
\int \nabla f_1(x)^\top f_2(x)\,dx
=
-\int f_1(x)\,\mathrm{div}(f_2)(x)\,dx,
\tag{111}
$$

其中 $f_1:\mathbb{R}^d\to\mathbb{R}, f_2:\mathbb{R}^d\to\mathbb{R}^d$；

以及

$$
\int f_1(x)\Delta f_2(x)\,dx
=
\int f_2(x)\Delta f_1(x)\,dx,
\tag{112}
$$

其中 $f_1,f_2:\mathbb{R}^d\to\mathbb{R}$。

### 从 SDE 的微小更新出发

使用正文里的随机更新形式（忽略高阶误差）：

$$
X_{t+h}
\approx
X_t + h\,u_t(X_t)+\sigma_t(W_{t+h}-W_t).
\tag{113}\tag{114}
$$

对测试函数 $f$ 做二阶 Taylor 展开：

$$
f(X_{t+h})-f(X_t)
\approx
\nabla f(X_t)^\top \bigl(h\,u_t(X_t)+\sigma_t(W_{t+h}-W_t)\bigr)
$$

$$
+\frac12
\bigl(h\,u_t(X_t)+\sigma_t(W_{t+h}-W_t)\bigr)^\top
\nabla^2 f(X_t)
\bigl(h\,u_t(X_t)+\sigma_t(W_{t+h}-W_t)\bigr).
$$

再对条件期望 $\mathbb{E}[\cdot\mid X_t]$ 取期望，利用

$$
W_{t+h}-W_t\mid X_t \sim \mathcal{N}(0,hI_d),
\qquad
\mathbb{E}[W_{t+h}-W_t\mid X_t]=0,
$$

以及

$$
\mathbb{E}_{\epsilon\sim \mathcal{N}(0,I_d)}[\epsilon^\top A\epsilon]
=
\mathrm{trace}(A),
$$

可得到

$$
\partial_t \mathbb{E}[f(X_t)]
=
\mathbb{E}\left[
\nabla f(X_t)^\top u_t(X_t)
+
\frac{\sigma_t^2}{2}\Delta f(X_t)
\right].
$$

又因为 $X_t\sim p_t$，这可以写成

$$
\partial_t \mathbb{E}[f(X_t)]
=
\int \nabla f(x)^\top u_t(x)p_t(x)\,dx
+
\frac{\sigma_t^2}{2}
\int \Delta f(x)p_t(x)\,dx.
$$

利用式 (111)、(112)，上式变为

$$
=
\int f(x)\left[
-\mathrm{div}(p_tu_t)(x)
+
\frac{\sigma_t^2}{2}\Delta p_t(x)
\right]dx.
$$

另一方面，

$$
\partial_t \mathbb{E}[f(X_t)]
=
\partial_t \int f(x)p_t(x)\,dx
=
\int f(x)\partial_t p_t(x)\,dx.
$$

由于这对任意测试函数 $f$ 都成立，结合式 (109)，就得到

$$
\partial_t p_t(x)
=
-\mathrm{div}(p_tu_t)(x)
+
\frac{\sigma_t^2}{2}\Delta p_t(x),
\qquad \forall x.
\tag{115}\tag{116}\tag{117}\tag{118}
$$

这就证明了 Fokker-Planck 方程是必要条件。

### 为什么它也是充分条件

Fokker-Planck 方程是一个偏微分方程，更具体地说，是一个抛物型 PDE。

与正文里的 ODE 唯一性类似，这类 PDE 在给定初值后也有唯一解。

如果某条路径 $p_t$ 满足式 (108)，而真实分布 $q_t$ 也满足同样的方程，并且两者初值相同：

$$
p_0=q_0=p_{\text{init}},
$$

那么由 PDE 解的唯一性就可知

$$
p_t=q_t,\qquad \forall 0\le t\le 1.
$$

也就是说，

$$
X_t\sim q_t=p_t.
$$

这就完成了充分性的证明。

## 附录 C 连续时间马尔可夫链的存在性与唯一性

这一节证明正文中的定理 33。

### 证明

#### 唯一性

我们要证明：满足式 (87) 的转移核

$$
p_{t'|t}(X_{t'}=y\mid X_t=x)
$$

只能有一个。

由式 (87) 可以推出

$$
\frac{d}{dt'}p_{t'|t}(X_{t'}=y\mid X_t=x)
=
\sum_{z\in\mathcal{S}} Q_{t'}(y\mid z)\,p_{t'|t}(X_{t'}=z\mid X_t=x).
\tag{119}\tag{120}\tag{121}\tag{122}
$$

对固定的 $x,t$ 而言，这就是一个关于

$$
t'\mapsto p_{t'|t}(X_{t'}=y\mid X_t=x)
$$

的线性 ODE。

它的初始条件是已知的：

$$
p_{t|t}(X_t=y\mid X_t=x)=\delta_y(x).
$$

由于线性 ODE 有唯一解，所以这个转移核也必须唯一。

#### 存在性

反过来，任何线性 ODE 都存在解。因此对每个 $x,t$，都存在一个函数

$$
p_{t'|t}(X_{t'}=y\mid X_t=x)
$$

满足

$$
p_{t|t}(X_t=y\mid X_t=x)=\delta_y(x),
\tag{123}
$$

以及

$$
\frac{d}{dt'}p_{t'|t}(X_{t'}=y\mid X_t=x)
=
\sum_{z\in\mathcal{S}} Q_{t'}(y\mid z)\,p_{t'|t}(X_{t'}=z\mid X_t=x).
\tag{124}
$$

对 $t'=t$，它自然就满足正文里的式 (87)。

剩下要做的是证明：这个解确实是一个合法的转移核，也就是它满足以下 3 条性质：

1. 概率和为 1

$$
\sum_{y\in\mathcal{S}} p_{t'|t}(X_{t'}=y\mid X_t=x)=1
\tag{125}
$$

2. 概率非负

$$
p_{t'|t}(X_{t'}=y\mid X_t=x)\ge 0
\tag{126}
$$

3. Chapman-Kolmogorov 条件

$$
\sum_{z\in\mathcal{S}}
p_{t_2|t_1}(X_{t_2}=y\mid X_{t_1}=z)
p_{t_1|t_0}(X_{t_1}=z\mid X_{t_0}=x)
=
p_{t_2|t_0}(y\mid x).
\tag{127}
$$

##### 性质 1：总和为 1

它在 $t'=t$ 时由式 (123) 立刻成立。

再看时间导数：

$$
\frac{d}{dt'}
\sum_{y\in\mathcal{S}} p_{t'|t}(X_{t'}=y\mid X_t=x)
=
\sum_{z\in\mathcal{S}}
\left[
\sum_{y\in\mathcal{S}}Q_{t'}(y\mid z)
\right]
p_{t'|t}(X_{t'}=z\mid X_t=x).
\tag{128}\tag{129}\tag{130}
$$

由于 rate matrix 的每一列和为 0，因此上式为 0：

$$
=0.
\tag{131}
$$

所以这个总和始终保持为 1。

##### 性质 2：非负性

在 $t'=t$ 时显然非负。

而且，当

$$
p_{t'|t}(X_{t'}=y\mid X_t=x)=0
$$

时，其导数满足

$$
\frac{d}{dt'}p_{t'|t}(X_{t'}=y\mid X_t=x)
=
\sum_{z\neq y} Q_{t'}(y\mid z)\,p_{t'|t}(X_{t'}=z\mid X_t=x)\ge 0.
$$

因为所有非对角项速率 $Q_{t'}(y\mid z)$ 都非负。

也就是说：一旦某个概率刚好到 0，它只能增加，不能继续变成负数。因此该核始终非负。

##### 性质 3：Chapman-Kolmogorov

定义

$$
q_{t_2|t_0}(y\mid x)
:=
\sum_{z\in\mathcal{S}}
p_{t_2|t_1}(X_{t_2}=y\mid X_{t_1}=z)\,
p_{t_1|t_0}(X_{t_1}=z\mid X_{t_0}=x).
$$

可以验证：

- 当 $t_2=t_1$ 时，

$$
q_{t_2=t_1|t_0}(y\mid x)=p_{t_1|t_0}(X_{t_1}=y\mid X_{t_0}=x)
$$

- 对 $t_2$ 求导，它满足与 $p_{t_2|t_0}(y\mid x)$ 完全相同的线性 ODE。

因此，由 ODE 解的唯一性可知

$$
q_{t_2|t_0}(y\mid x)=p_{t_2|t_0}(y\mid x).
$$

于是 Chapman-Kolmogorov 条件成立。

综上，$p_{t'|t}(y\mid x)$ 的确是一个合法转移核，也就完成了定理 33 的证明。

## 附录 D 关于 VAE 的更多视角

这一节在正文基础上进一步展开 VAE 的讨论，并给出正文式 (83) 中总 VAE 损失的变分推导。

第一步，注意 encoder 和 decoder 都会诱导出定义在数据 $x$ 和 latent $z$ 上的联合分布：

$$
q_\phi(x,z)=p_{\text{data}}(x)\,q_\phi(z\mid x)
\qquad\text{（encoder 联合分布）}
$$

$$
p_\theta(x,z)=p_\theta(x\mid z)\,p_{\text{prior}}(z)
\qquad\text{（decoder 联合分布）}
$$

因此，我们可以把训练 VAE 理解为：学习 $\phi,\theta$，让这两个联合分布彼此尽量接近。

最自然的度量就是联合分布上的 KL 散度：

$$
D_{\mathrm{KL}}(q_\phi(x,z)\|p_\theta(x,z))
=
D_{\mathrm{KL}}(p_{\text{data}}(x)q_\phi(z\mid x)\|p_\theta(x\mid z)p_{\text{prior}}(z)).
\tag{132}
$$

展开后可得三部分：

1. $\mathbb{E}[\log p_{\text{data}}(x)]$：这是一个与 $\phi,\theta$ 无关的常数

$$
\mathbb{E}[\log p_{\text{data}}(x)] = C
\tag{133}
$$

2. 一项 prior regularization：

$$
\mathbb{E}_{x\sim p_{\text{data}}}
\bigl[
D_{\mathrm{KL}}(q_\phi(z\mid x)\|p_{\text{prior}}(z))
\bigr]
\tag{134}
$$

3. 一项 reconstruction term：

$$
-\mathbb{E}_{x\sim p_{\text{data}}(x),\,z\sim q_\phi(z\mid x)}
[\log p_\theta(x\mid z)].
\tag{135}
$$

因此，忽略常数项后，VAE 损失其实就是联合空间中的 KL 散度：

$$
L_{\text{VAE}}(\phi,\theta)
=
\mathbb{E}_{x\sim p_{\text{data}}}
\bigl[
D_{\mathrm{KL}}(q_\phi(z\mid x)\|p_{\text{prior}}(z))
\bigr]
-
\mathbb{E}_{x,z}
[\log p_\theta(x\mid z)]
$$

$$
=
D_{\mathrm{KL}}(q_\phi(x,z)\|p_\theta(x,z))
+
\text{const}.
\tag{136}\tag{137}
$$

### 把 VAE 看成生成模型

我们也可以把 VAE 本身看成一个生成模型：

1. 先采样

$$
z\sim p_{\text{prior}}=\mathcal{N}(0,I_k)
$$

2. 再从 decoder 中采样

$$
x\sim p_\theta(\cdot\mid z)
$$

于是最终生成分布为

$$
p_\theta(x)=\int p_\theta(x\mid z)p_{\text{prior}}(z)\,dz.
$$

接下来需要用到一个链式规则。

### 命题 3（KL 的链式规则）

设 $q(x,z),p(x,z)$ 是定义在两个变量上的分布，则

$$
D_{\mathrm{KL}}(q(z,x)\|p(z,x))
=
D_{\mathrm{KL}}(q(x)\|p(x))
+
\mathbb{E}_{x\sim q}
\bigl[
D_{\mathrm{KL}}(q(z\mid x)\|p(z\mid x))
\bigr].
$$

特别地，由于第二项总是非负，就得到数据处理不等式：

$$
D_{\mathrm{KL}}(q(x)\|p(x))
\le
D_{\mathrm{KL}}(q(z,x)\|p(z,x)).
\tag{138}
$$

### 两个重要上界

由命题 3 可以推出

$$
L_{\text{VAE}}(\phi,\theta)
=
D_{\mathrm{KL}}(q_\phi(x,z)\|p_\theta(x,z))
+
\text{const}
\ge
D_{\mathrm{KL}}(p_{\text{data}}(x)\|p_\theta(x))
+
\text{const}.
\tag{139}
$$

也就是说，VAE 实际上是在最小化一个上界，这个上界约束了生成分布 $p_\theta(x)$ 和真实数据分布 $p_{\text{data}}(x)$ 之间的 KL 距离。

同样地，也有

$$
L_{\text{VAE}}(\phi,\theta)
\ge
D_{\mathrm{KL}}(q_\phi(z)\|p_{\text{prior}}(z))
+
\text{const}.
\tag{140}
$$

也就是说，VAE 目标也在最小化 latent 分布和 prior 之间 KL 距离的一个上界。

### 为什么还需要单独训练 latent 生成模型？

既然 VAE 本身就可以生成样本，那为什么还要在 latent space 里再训练一个 flow / diffusion model？

原因在于 amortization gap。虽然

$$
D_{\mathrm{KL}}(q_\phi(x,z)\|p_\theta(x,z))
$$

变小会推动

$$
D_{\mathrm{KL}}(q_\phi(z)\|p_{\text{prior}}(z))
$$

变小，但二者并不会完全同步减小。

对应的 gap 可以写成

$$
D_{\mathrm{KL}}(q_\phi(x,z)\|p_\theta(x,z))
-
D_{\mathrm{KL}}(q_\phi(z)\|p_{\text{prior}}(z)).
\tag{141}
$$

因此训练结束时，通常仍有

$$
q_\phi(z)\neq p_{\text{prior}}(z).
$$

而 decoder 在训练时看到的是来自 $q_\phi(z)$ 的 latent，而不是来自 prior 的 latent，所以推理时如果直接从 prior 采样，有一定 out-of-distribution 风险。

但实践中，这反而常常是优点：因为 flow / diffusion model 通常比 VAE decoder 更强，所以把更多“生成复杂度”交给 latent generative model 往往更合理。

### ELBO 视角

对固定的 $x$，有

$$
\mathbb{E}_{z\sim q_\phi(z\mid x)}
\log \frac{q_\phi(z\mid x)}{p_\theta(x\mid z)p_{\text{prior}}(z)}
=
D_{\mathrm{KL}}(q_\phi(z\mid x)\|p_\theta(z\mid x))
-\log p_\theta(x).
\tag{142}
$$

移项可得

$$
\mathbb{E}_{z\sim q_\phi(z\mid x)}
\log \frac{p_\theta(x\mid z)p_{\text{prior}}(z)}{q_\phi(z\mid x)}
+
D_{\mathrm{KL}}(q_\phi(z\mid x)\|p_\theta(z\mid x))
=
\log p_\theta(x).
\tag{143}
$$

于是

$$
\underbrace{
\mathbb{E}_{z\sim q_\phi(z\mid x)}
\log \frac{p_\theta(x\mid z)p_{\text{prior}}(z)}{q_\phi(z\mid x)}
}_{\text{ELBO}(x;\phi,\theta)}
\le
\log p_\theta(x).
\tag{144}
$$

这个左边就叫 evidence lower bound（ELBO）。

进一步可以把 VAE 损失写成

$$
L_{\text{VAE}}
=
-\mathbb{E}_{x\sim p_{\text{data}}}[\mathrm{ELBO}(x;\phi,\theta)]
+
\text{const}.
\tag{145}
$$

所以，VAE 训练也可以理解为：

> 最大化数据上的期望 ELBO。

### 注释 42（当 $q_\phi(x,z)\approx p_\theta(x,z)$ 时会发生什么？）

如果

$$
q_\phi(x,z)\approx p_\theta(x,z),
$$

那么：

1. latent 采样分布

$$
q_\phi(z)=\int q_\phi(z\mid x)p_{\text{data}}(x)\,dx
$$

会接近

$$
p_\theta(z)=p_{\text{prior}}(z),
$$

即 latent 分布被很好地正则化了。

2. 同时也意味着变分近似较好：

$$
q_\phi(z\mid x)\approx p_\theta(z\mid x),
$$

从而 reconstruction error 会比较低。

### 注释 43（VAE 里“变分”的含义）

为什么不直接让

$$
q_\phi(\cdot\mid x)=p_\theta(\cdot\mid x)
$$

呢？

因为虽然我们知道 likelihood $p_\theta(x\mid z)$，但 posterior

$$
p_\theta(z\mid x)
=
\frac{p_\theta(x\mid z)p_{\text{prior}}(z)}{p_\theta(x)}
$$

通常是不可 tractable 的，因为分母 $p_\theta(x)$ 本身算不出来。

因此，VAE 中的 “variational” 就体现在：

$$
q_\phi(\cdot\mid x)
$$

只是对真实后验

$$
p_\theta(\cdot\mid x)
$$

的一个可学习近似。

### Reconstruction vs Generation

给定 encoder $q_\phi(z\mid x)$、decoder $p_\theta(x\mid z)$，以及学习去逼近 $q_\phi(z)$ 的 latent generative model $r_\psi$，可以定义两类采样器：

1. reconstruction sampler：

$$
r_{\psi,\theta}^{\text{recon}}(x_{\text{out}})
=
\iint
p_\theta(x_{\text{out}}\mid z)\,
q_\phi(z\mid x_{\text{data}})\,
p_{\text{data}}(x_{\text{data}})
\,dz\,dx_{\text{in}}
$$

2. generative sampler：

$$
r_{\psi,\phi}^{\text{gen}}(x_{\text{out}})
=
\int p_\theta(x_{\text{out}}\mid z_{\text{gen}})\,r_\psi(z_{\text{gen}})\,dz_{\text{gen}}.
$$

前者是“先从真实数据编码，再解码”，后者是“先从 latent 生成模型采样，再解码”。

讲义指出：重建质量（例如 rFID）和真正生成质量（例如 gFID）之间通常存在张力。高质量重建通常意味着 latent 中保留了很多信息，从而 latent 分布更复杂、更难学；而更强压缩则让 latent 更简单、但 reconstruction 会变差。

### Intuition 44（劳动分工）

Figure 22 里最重要的直觉是：

> autoencoder 与 latent generative model 之间存在一个“最优分工点”，大约位于 Pareto frontier 的膝盖处。

在那个点上：

- 压缩已经足够强，因此 latent generative model 更容易训练；
- 同时 reconstruction distortion 还没有高到不可接受。

这正是现代 latent diffusion 系统设计里非常核心的工程直觉。

## 附录 E Diffusion Model 文献导读

这一节不是正文必要内容，而是帮助你在阅读相关论文时少踩坑。因为文献里的 diffusion model / flow matching 有很多表述方式，和本讲义不完全一样，但其实大多是等价的。

### 离散时间 vs 连续时间

最早的一批 denoising diffusion model 论文 [41, 42, 17] 用的不是 SDE，而是离散时间马尔可夫链，也就是时间步

$$
t=0,1,2,3,\ldots
$$

这种离散形式。

它的优点是简单；但缺点是：

1. 训练前必须先固定时间离散方式；
2. 损失函数通常只能通过 ELBO 来近似，也就是说优化的是一个下界，而不是我们真正想优化的目标。

后来 Song 等人 [45] 说明：这些离散构造本质上是在逼近连续时间的 SDE。

而且在连续时间情形下，ELBO 会变成严格等式。比如本讲义里的定理 12 和定理 22 都是“相等”，不是“下界”。

因此，SDE 形式后来变得更流行，因为它被认为数学上更干净，也允许在训练后通过 ODE/SDE sampler 去灵活控制采样误差。

不过本质上，离散时间和连续时间模型用的核心损失并没有根本不同。

### “Forward process” vs probability path

第一代 diffusion model [41, 42, 17, 45] 通常不会用“probability path”这个词，而是通过一个所谓的 forward process 来对数据点 $z\in\mathbb{R}^d$ 加噪：

$$
\bar X_0 = z,\qquad
d\bar X_t = u_t^{\text{forw}}(\bar X_t)\,dt+\sigma_t^{\text{forw}}\,d\bar W_t.
\tag{146}
$$

它的含义是：先采样一个数据点 $z\sim p_{\text{data}}$，再通过这个 forward process 逐渐把它“腐蚀”成噪声。这个过程被设计成：

$$
\bar X_T\approx \mathcal{N}(0,I_d)
\qquad \text{当 }T\gg 0.
$$

实际上，这就对应一条概率路径：

- 条件分布 $\bar X_t\mid \bar X_0=z$ 给出一条条件概率路径；
- 再对 $z\sim p_{\text{data}}$ 做边缘化，就得到边缘概率路径。

不过，forward process 有个限制：为了避免训练时真的去模拟 SDE，我们必须知道

$$
\bar X_t\mid \bar X_0=z
$$

的闭式表达。

这会把 forward process 的向量场限制在仿射形式上：

$$
u_t^{\text{forw}}(x)=a_t x.
$$

对这种形式，可以推导出

$$
\bar X_t\mid \bar X_0=z
\sim
\mathcal{N}(\alpha_t z,\beta_t^2 I),
$$

其中

$$
\alpha_t
=
\exp\left(\int_0^t a_r\,dr\right),
\qquad
\beta_t^2
=
\alpha_t^2
\int_0^t
\frac{(\sigma_r^{\text{forw}})^2}{\alpha_r^2}\,dr.
$$

也就是说：

> forward process 本质上只是构造高斯 probability path 的一种特殊方式。

而 flow matching [25] 引入 “probability path” 这个表述，正是为了让框架更简单、更一般。

### 时间反演 vs 直接解 Fokker-Planck

最早 diffusion model 的训练目标，并不是像本讲义这样通过 Fokker-Planck 方程 / 连续性方程得到的，而是通过对 forward process 做时间反演（time reversal）得到 [2]。

所谓时间反演，是指构造另一个随机过程 $(X_t)_{0\le t\le T}$，使得它在时间上反向复现原过程的轨迹分布：

$$
\mathbb{P}[\bar X_{t_1}\in A_1,\ldots,\bar X_{t_n}\in A_n]
=
\mathbb{P}[X_{T-t_1}\in A_1,\ldots,X_{T-t_n}\in A_n].
\tag{147}\tag{148}
$$

Anderson [2] 证明了，满足这一性质的时间反演过程可以写成

$$
dX_t
=
\left(
-u_t(X_t)+\sigma_t^2 \nabla \log p_t(X_t)
\right)dt
+
\sigma_t dW_t,
$$

其中 $u_t,\sigma_t$ 与 forward process 对应。

由于 diffusion model 最终往往只关心末端生成样本，而不会保留整条轨迹，所以：

> 一个过程是否真的是“严格时间反演”，在很多生成任务里并不重要。

因此，如今很多人更倾向于直接通过 Fokker-Planck 方程来构造训练目标，这也是本讲义采用的路线。

### Flow Matching 与 Stochastic Interpolants

本讲义的表述方式，与 flow matching 和 stochastic interpolants（SI）最接近。

其中：

- flow matching 只关注 flow，也就是说采样本身是确定性的（随机性只来自初始点 $X_0\sim p_{\text{init}}$）；
- stochastic interpolants 同时包含纯 flow 和正文里第 4 节讲到的 SDE 扩展。

stochastic interpolants 这个名字来自它所使用的插值函数 $I(t,x,z)$，它本质上是构造条件/边缘 probability path 的另一种方式。

与传统 diffusion model 相比，flow matching / SI 的优势在于：

1. 训练框架更简单；
2. 更一般；
3. 可以处理任意初始分布 $p_{\text{init}}$ 与任意目标分布 $p_{\text{data}}$。

而经典 denoising diffusion model 通常只针对高斯初始分布和高斯 probability path。

### 小结 45（文献中的其他 diffusion 表述）

在文献里，常见的 diffusion model 替代表述通常包含下面这些元素中的若干个：

1. 离散时间：很多工作会把 SDE 近似成离散时间马尔可夫链。
2. 反向时间约定：经常用与本讲义相反的时间约定，也就是 $t=0$ 对应 $p_{\text{data}}$。
3. Forward process：本质上是构造高斯 probability path 的一种方式。
4. 通过时间反演得到训练目标：这是本讲义中构造方式的一个特殊实例，只是采用了反向时间记号。

因此，当你去读 diffusion model / flow matching 文献时，虽然记号和表述可能看起来和这份讲义不一样，但很多时候，它们本质上讲的是同一件事。
