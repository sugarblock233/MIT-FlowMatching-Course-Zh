# 第 4 节 Score Functions 与 Score Matching（中文翻译）

对应原讲义 `lecture_notes.pdf` 第 25-33 页，即 Section 4：`Score Functions and Score Matching`，包含 4.1、4.2、4.3 三个小节。

## 4 Score Functions 与 Score Matching

上一节我们讲了如何用 flow matching 来训练 flow model。这一节我们转向 diffusion model，并说明如何使用 score matching 来训练它们。

## 4.1 条件与边缘 Score Function

到目前为止，我们研究的核心对象一直是向量场 $u_t(x)$。而 diffusion model [45, 44] 采用了另一种视角：它们更关注 score function。

因此，这一节里，我们会用 score function 的语言，把前面学过的内容重新表述一遍。这会给我们一个新的理解角度。

设 $q(x)$ 是任意一个概率分布。则 $q$ 的 score function 定义为

$$
\nabla \log q(x),
$$

也就是对数似然 $\log q(x)$ 关于 $x$ 的梯度。

这个量有很直观的含义：$\nabla \log q(x)$ 给出了让对数似然增长最快的方向，也就是“最陡上升方向”。

现在回到第 3 节中的条件概率路径 $p_t(x\mid z)$ 和边缘概率路径 $p_t(x)$。

于是我们可以定义：

- 条件 score function：

$$
\nabla \log p_t(x\mid z)
$$

- 边缘 score function：

$$
\nabla \log p_t(x)
$$

与前面的边缘向量场公式（式 (18)）类似，边缘 score 也可以由条件 score 表示为

$$
\nabla \log p_t(x)
=
\int
\nabla \log p_t(x\mid z)
\frac{p_t(x\mid z)p_{\text{data}}(z)}{p_t(x)}
\,dz.
\tag{38}
$$

因此，条件 score 与边缘 score 之间的关系，和条件向量场与边缘向量场之间的关系完全类似。

这个式子可以直接证明：

$$
\nabla \log p_t(x)
=
\frac{\nabla p_t(x)}{p_t(x)}
=
\frac{\nabla \int p_t(x\mid z)p_{\text{data}}(z)\,dz}{p_t(x)}
$$

$$
=
\frac{\int \nabla p_t(x\mid z)p_{\text{data}}(z)\,dz}{p_t(x)}
=
\int
\nabla \log p_t(x\mid z)\frac{p_t(x\mid z)p_{\text{data}}(z)}{p_t(x)}\,dz.
\tag{39}
$$

这里两次都用到了

$$
\partial_y \log y = \frac{1}{y}
$$

再配合链式法则。

### 例 15（高斯概率路径的 Score Function）

对于高斯路径

$$
p_t(x\mid z)=\mathcal{N}(x;\alpha_t z,\beta_t^2 I_d),
$$

利用高斯密度的形式，可以得到

$$
\nabla \log p_t(x\mid z)
=
\nabla \log \mathcal{N}(x;\alpha_t z,\beta_t^2 I_d)
=
-\frac{x-\alpha_t z}{\beta_t^2}.
\tag{40}
$$

注意：高斯概率路径的 score function 关于 $x$ 和 $z$ 都是线性的。而前面第 3 节里的条件向量场 $u_t(x\mid z)$（见式 (20)）也是线性的。

因此，两者之间可以互相转换。下面这个命题把这个关系说清楚了。

### 命题 1（高斯概率路径下的转换公式）

对于高斯概率路径

$$
p_t(x\mid z)=\mathcal{N}(\alpha_t z,\beta_t^2 I_d),
$$

条件/边缘向量场与条件/边缘 score 之间满足如下关系：

$$
u_t^{\text{target}}(x\mid z)
=
a_t \nabla \log p_t(x\mid z)+b_t x,
\qquad
a_t=\beta_t^2\left(\frac{\dot\alpha_t}{\alpha_t}-\frac{\dot\beta_t}{\beta_t}\right),
\qquad
b_t=\frac{\dot\alpha_t}{\alpha_t},
\tag{41}
$$

以及

$$
u_t^{\text{target}}(x)
=
a_t \nabla \log p_t(x)+b_t x.
\tag{42}
$$

这说明：

- 给定条件 score，可以恢复条件向量场；
- 给定边缘 score，可以恢复边缘向量场；
- 反过来也成立。

#### 证明

先看条件向量场。由第 3 节推导出的公式：

$$
u_t^{\text{target}}(x\mid z)
=
\left(
\dot\alpha_t-\frac{\dot\beta_t}{\beta_t}\alpha_t
\right)z
+
\frac{\dot\beta_t}{\beta_t}x.
$$

做一点代数变形，可得

$$
u_t^{\text{target}}(x\mid z)
=
\beta_t^2
\left(
\frac{\dot\alpha_t}{\alpha_t}-\frac{\dot\beta_t}{\beta_t}
\right)
\frac{\alpha_t z-x}{\beta_t^2}
+
\frac{\dot\alpha_t}{\alpha_t}x.
$$

注意到

$$
\frac{\alpha_t z-x}{\beta_t^2}
=
\nabla \log p_t(x\mid z),
$$

于是就得到

$$
u_t^{\text{target}}(x\mid z)
=
a_t\nabla \log p_t(x\mid z)+b_t x.
$$

再对两边取关于后验分布的积分，就得到边缘版本：

$$
u_t^{\text{target}}(x)
=
\int u_t^{\text{target}}(x\mid z)\frac{p_t(x\mid z)p_{\text{data}}(z)}{p_t(x)}\,dz
=
a_t\nabla \log p_t(x)+b_t x,
$$

其中用到了式 (38) 以及后验密度积分为 1 的事实。

命题 1 很重要，因为它说明：一旦我们学到了 $u_t^{\text{target}}$，其实也就等价于学到了 score function $\nabla \log p_t(x)$；反过来也一样。

因此，很多 diffusion model 会直接用神经网络去学习 score function $\nabla \log p_t(x)$。我们会在 4.3 节讨论这一点。

### 注释 16（Score 的重参数化）

高斯概率路径下，式 (41) 这样的重参数化之所以成立，是因为条件向量场和条件 score 都是关于 $x,z$ 的线性函数。

当我们把它们边缘化之后，两边其实都只是后验均值

$$
\mathbb{E}_{z\mid x}[z]
$$

的一个线性重参数化。

因此，任何能够恢复 $\mathbb{E}_{z\mid x}[z]$ 的量，都可以用来恢复无条件向量场和无条件 score。并且，从数值稳定性或训练稳定性的角度出发，这么做甚至可能更合适。

一个常见选择就是后验均值本身，人们通常把它称为 denoiser（去噪器）。形式上，条件和边缘 denoiser 定义为

$$
D_t(x\mid z)=z,
$$

以及

$$
D_t(x)
=
\int z\frac{p_t(x\mid z)p_{\text{data}}(z)}{p_t(x)}\,dz.
\tag{43}
$$

它的直观含义非常清楚：

> 在给定噪声数据 $x$ 时，对“干净数据” $z$ 的条件期望。

很多模型之所以叫 denoising diffusion models，就是因为学习 $D_t$ 与学习 $u_t^{\text{target}}$ 在理论上是等价的。

## 4.2 用 SDE 进行采样

到目前为止，我们已经说明了：如何构造一个 ODE 轨迹 $X_t$，使其通过边缘向量场 $u_t^{\text{target}}$ 沿着给定概率路径 $p_t$ 演化。

但这种做法只适用于 flow model。那 diffusion model 呢？

现在我们借助 score function，把这个结果扩展到 SDE。

### 定理 17（SDE 扩展技巧）

仍然像前面那样定义条件向量场和边缘向量场 $u_t^{\text{target}}(x\mid z)$ 与 $u_t^{\text{target}}(x)$。那么，对任意扩散系数 $\sigma_t\ge 0$，我们都可以在原始 ODE 的动力学上加入随机项，从而构造下面这个 SDE：

$$
X_0\sim p_{\text{init}},\qquad
dX_t
=
u_t^{\text{target}}(X_t)\,dt
+
\frac{\sigma_t^2}{2}\nabla \log p_t(X_t)\,dt
+
\sigma_t\,dW_t.
\tag{44}
$$

则它满足

$$
X_t\sim p_t,\qquad 0\le t\le 1.
\tag{45}
$$

特别地，

$$
X_1\sim p_{\text{data}}.
$$

这里加入的随机动力学与 Langevin dynamics 很接近。你可以把它理解为：

> 一边注入噪声，一边保持边缘分布 $p_t$ 不变。

图 9 展示了定理 17 的效果。和 ODE 情形相比，现在轨迹会呈现明显的“之字形”，这正体现了 SDE 的随机性；但尽管轨迹变随机了，边缘分布 $p_t$ 却保持不变。

这个结果有一个非常 striking 的地方：

> 即使网络已经训练好了，我们仍然可以在理论上任意选择扩散系数 $\sigma_t\ge 0$。

从理论上讲，定理 17 对所有这样的 $\sigma_t$ 都成立。

不过在实践中，我们会遇到两种误差：

1. 训练误差：神经网络并没有完美逼近真实的边缘向量场和 score；
2. 数值模拟误差：比如当 $\sigma_t$ 很大时，算法 2 可能需要非常小的步长才能稳定工作。

因此在实际中，对一个固定的已训练模型，往往存在一个经验上最优的 $\sigma_t\ge 0$，可以通过实验调出来 [23, 1, 28]。

### 例 18（高斯情形下的 SDE 扩展技巧）

对于高斯概率路径，根据命题 1，我们已经能从边缘向量场直接得到 score function。因此，定理 17 里的 SDE 可以纯粹用 score 写成：

$$
X_0\sim p_{\text{init}},\qquad
dX_t
=
\left(
a_t+\frac{\sigma_t^2}{2}
\right)\nabla \log p_t(X_t)\,dt
+
b_t X_t\,dt
+
\sigma_t\,dW_t,
\tag{46}
$$

并满足

$$
X_t\sim p_t,\qquad 0\le t\le 1.
\tag{47}
$$

其中 $a_t,b_t$ 的定义与命题 1 相同。

接下来，讲义用 Fokker-Planck 方程来证明定理 17。这个方程可以看作连续性方程在 SDE 场景下的推广。

先定义 Laplacian（拉普拉斯算子）：

$$
\Delta w_t(x)
=
\sum_{i=1}^d \frac{\partial^2}{\partial x_i^2}w_t(x)
=
\mathrm{div}(\nabla w_t)(x),
\tag{48}
$$

其中 $w_t:\mathbb{R}^d\to \mathbb{R}$ 是一个标量场。

### 定理 19（Fokker-Planck 方程）

设 $p_t$ 是一条概率路径，考虑 SDE

$$
X_0\sim p_{\text{init}},\qquad
dX_t = u_t(X_t)\,dt+\sigma_t\,dW_t.
$$

则对所有 $0\le t\le 1$，$X_t$ 服从分布 $p_t$ 当且仅当 Fokker-Planck 方程成立：

$$
\partial_t p_t(x)
=
-\mathrm{div}(p_tu_t)(x)
+
\frac{\sigma_t^2}{2}\Delta p_t(x),
\qquad
\forall x\in\mathbb{R}^d,\ 0\le t\le 1.
\tag{49}
$$

当 $\sigma_t=0$ 时，这个方程就退化成前面第 3 节里的连续性方程。

那个额外出现的拉普拉斯项 $\Delta p_t$ 一开始可能不太直观，但熟悉物理的同学会注意到：它和热方程里的项是一样的。热在介质中会扩散；这里我们也加入了一个扩散过程，因此方程中自然多出了这个拉普拉斯项。

### 定理 17 的证明

根据定理 19，我们只需要证明式 (44) 中定义的 SDE 满足 $p_t$ 的 Fokker-Planck 方程即可。

直接计算：

$$
\partial_t p_t(x)
=
-\mathrm{div}(p_tu_t^{\text{target}})(x)
$$

给右边加上再减去同一个项：

$$
=
-\mathrm{div}(p_tu_t^{\text{target}})(x)
-\frac{\sigma_t^2}{2}\Delta p_t(x)
+\frac{\sigma_t^2}{2}\Delta p_t(x)
$$

再利用

$$
\Delta p_t = \mathrm{div}(\nabla p_t)
$$

可写成

$$
=
-\mathrm{div}(p_tu_t^{\text{target}})(x)
-\mathrm{div}\left(\frac{\sigma_t^2}{2}\nabla p_t\right)(x)
+\frac{\sigma_t^2}{2}\Delta p_t(x).
$$

再用

$$
\nabla \log p_t = \frac{\nabla p_t}{p_t},
$$

得到

$$
=
-\mathrm{div}(p_tu_t^{\text{target}})(x)
-\mathrm{div}\left(
p_t\frac{\sigma_t^2}{2}\nabla \log p_t
\right)(x)
+\frac{\sigma_t^2}{2}\Delta p_t(x).
$$

合并后就是

$$
=
-\mathrm{div}\left(
p_t\left(
u_t^{\text{target}}+\frac{\sigma_t^2}{2}\nabla \log p_t
\right)
\right)(x)
+\frac{\sigma_t^2}{2}\Delta p_t(x).
$$

这正是式 (44) 对应的 Fokker-Planck 方程，因此由定理 19 可知

$$
X_t\sim p_t,\qquad 0\le t\le 1.
$$

证毕。

### 注释 20（可选：Langevin Dynamics）

上面的构造有一个非常著名的特例：当概率路径是常数路径时，也就是

$$
p_t=p
$$

对所有 $t$ 都成立。

这时令

$$
u_t^{\text{target}}=0,
$$

则得到 SDE

$$
dX_t = \frac{\sigma_t^2}{2}\nabla \log p(X_t)\,dt+\sigma_t\,dW_t.
\tag{50}
$$

这就是人们熟知的 Langevin dynamics。

由于现在 $p_t$ 不随时间变化，所以

$$
\partial_t p_t(x)=0.
$$

由定理 17 立刻可以知道，$p$ 是 Langevin dynamics 的平稳分布（stationary distribution）：

$$
X_0\sim p
\quad\Longrightarrow\quad
X_t\sim p,\qquad t\ge 0.
$$

并且像许多马尔可夫过程一样，在相当一般的条件下，它还会收敛到这个平稳分布。也就是说，如果初值不是从 $p$ 来，而是

$$
X_0\sim p'\neq p,
$$

则在较弱条件下，分布 $p_t'$ 会逐渐收敛到 $p$。

这使 Langevin dynamics 在很多领域都非常有用，例如分子动力学模拟，以及 Bayesian 统计和自然科学中的各种 MCMC 方法。

特别地，当 $p$ 是一个高斯分布时，就会退化成 Ornstein-Uhlenbeck 过程，而这正是早期 diffusion model 的基础之一。

### 注释 21（可选：GLASS Flows）

相比 ODE，SDE 采样的一个显著特点是：演化变成了随机的，也就是说，初始点 $X_0$ 不再唯一决定 $t>0$ 时的 $X_t$。

有趣的是，最近有人发现，通过一个叫做 GLASS Flows [20] 的简单采样技巧，即使只用 ODE，也可以实现相同的随机转移效果。

这意味着：我们可以利用 SDE 那种“演化带有随机性”的优点（例如配合搜索算法），同时保留 ODE 在计算效率上的优势。

## 4.3 Score Matching

现在还剩最后一个问题：

> 我们怎样学习边缘 score function $\nabla \log p_t(x)$？

当然，在高斯概率路径下，我们可以直接用命题 1，把已经学到的 $u_t^{\text{target}}(x)$ 转换成 score。

但是在一般情形下怎么办？

答案是：我们也可以直接学习边缘 score function 本身。

为了逼近边缘 score $\nabla \log p_t$，我们引入一个神经网络，称为 score network：

$$
s_t^\theta:\mathbb{R}^d\times [0,1]\to \mathbb{R}^d.
$$

和前面一样，我们可以定义两种损失：

- score matching loss

$$
L_{\mathrm{SM}}(\theta)
=
\mathbb{E}_{t\sim \mathrm{Unif},\,z\sim p_{\text{data}},\,x\sim p_t(\cdot\mid z)}
\bigl[
\|s_t^\theta(x)-\nabla \log p_t(x)\|^2
\bigr]
$$

- conditional / denoising score matching loss

$$
L_{\mathrm{CSM}}(\theta)
=
\mathbb{E}_{t\sim \mathrm{Unif},\,z\sim p_{\text{data}},\,x\sim p_t(\cdot\mid z)}
\bigl[
\|s_t^\theta(x)-\nabla \log p_t(x\mid z)\|^2
\bigr].
$$

两者的区别，和前面一模一样：

- 前者使用的是边缘 score $\nabla \log p_t(x)$
- 后者使用的是条件 score $\nabla \log p_t(x\mid z)$

理想情况下，我们当然想最小化 $L_{\mathrm{SM}}$，但因为我们并不知道 $\nabla \log p_t(x)$，它并不可计算。

不过和前面一样，$L_{\mathrm{CSM}}$ 是一个 tractable 的替代目标。

### 定理 22

Score matching loss 与 denoising score matching loss 只差一个常数：

$$
L_{\mathrm{SM}}(\theta)=L_{\mathrm{CSM}}(\theta)+C,
$$

其中 $C$ 与参数 $\theta$ 无关。

因此，它们的梯度相同：

$$
\nabla_\theta L_{\mathrm{SM}}(\theta)
=
\nabla_\theta L_{\mathrm{CSM}}(\theta).
$$

特别地，对于最优解 $\theta^\*$，有

$$
s_t^{\theta^\*}=\nabla \log p_t.
$$

#### 证明

这个证明和第 3 节定理 12 的证明完全一样。因为边缘 score 的公式（式 (38)）与边缘向量场的公式（式 (18)）在形式上是相同的，所以只要把其中的 $u_t^{\text{target}}$ 全部替换成 $\nabla \log p_t$ 即可。

### 例 23（高斯概率路径下的 Denoising Score Matching）

现在把 denoising score matching loss 具体化到高斯路径

$$
p_t(x\mid z)=\mathcal{N}(\alpha_t z,\beta_t^2 I_d).
$$

由式 (40)，条件 score 为

$$
\nabla \log p_t(x\mid z)
=
-\frac{x-\alpha_t z}{\beta_t^2}.
\tag{51}
$$

把这个公式代入条件 score matching loss，可得

$$
L_{\mathrm{CSM}}(\theta)
=
\mathbb{E}_{t\sim \mathrm{Unif},\,z\sim p_{\text{data}},\,x\sim p_t(\cdot\mid z)}
\left[
\left\|
s_t^\theta(x)+\frac{x-\alpha_t z}{\beta_t^2}
\right\|^2
\right].
$$

再把

$$
x=\alpha_t z+\beta_t \epsilon,\qquad
\epsilon\sim \mathcal{N}(0,I_d)
$$

代入，就得到

$$
L_{\mathrm{CSM}}(\theta)
=
\mathbb{E}_{t\sim \mathrm{Unif},\,z\sim p_{\text{data}},\,\epsilon\sim \mathcal{N}(0,I_d)}
\left[
\left\|
s_t^\theta(\alpha_t z+\beta_t\epsilon)+\frac{\epsilon}{\beta_t}
\right\|^2
\right]
$$

等价地，也可以写成

$$
=
\mathbb{E}_{t\sim \mathrm{Unif},\,z\sim p_{\text{data}},\,\epsilon\sim \mathcal{N}(0,I_d)}
\left[
\frac{1}{\beta_t^2}
\left\|
\beta_t s_t^\theta(\alpha_t z+\beta_t\epsilon)+\epsilon
\right\|^2
\right].
$$

注意，这里网络 $s_t^\theta$ 本质上是在学习“预测用来污染数据样本 $z$ 的噪声”。这也是为什么这个训练目标叫 denoising score matching。

后来人们发现：当 $\beta_t\approx 0$ 时，上面的损失在数值上不太稳定，也就是说，当噪声特别小时，训练会变难。

因此，在最早的一些 denoising diffusion model 工作中（例如 DDPM [17]），人们提出：

1. 直接去掉损失中的常数因子 $1/\beta_t^2$
2. 把 score network $s_t^\theta$ 重参数化成一个噪声预测网络 $\epsilon_t^\theta$：

$$
-\beta_t s_t^\theta(x)=\epsilon_t^\theta(x).
$$

于是损失变成

$$
L_{\mathrm{DDPM}}(\theta)
=
\mathbb{E}_{t\sim \mathrm{Unif},\,z\sim p_{\text{data}},\,\epsilon\sim \mathcal{N}(0,I_d)}
\bigl[
\|\epsilon_t^\theta(\alpha_t z+\beta_t\epsilon)-\epsilon\|^2
\bigr].
$$

和前面一样，这个网络本质上就是在预测把数据样本 $z$ 污染后的噪声 $\epsilon$。

### 算法 4：高斯概率路径下的 Score Matching 训练流程

输入：样本数据集 $z\sim p_{\text{data}}$，score network $s_t^\theta$ 或 noise predictor $\epsilon_t^\theta$

1. 对每个 mini-batch：
2. 从数据集中采样一个样本 $z$
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
x_t=\alpha_t z+\beta_t \epsilon
\qquad
\text{（一般情形下： }x_t\sim p_t(\cdot\mid z)\text{）}
$$

6. 计算损失：

如果使用 score network：

$$
L(\theta)=
\left\|
s_t^\theta(x_t)+\frac{\epsilon}{\beta_t}
\right\|^2
$$

如果使用 noise predictor：

$$
L(\theta)=\|\epsilon_t^\theta(x_t)-\epsilon\|^2
$$

一般情形下，可统一写为

$$
\|s_t^\theta(x_t)-\nabla \log p_t(x_t\mid z)\|^2.
$$

7. 用梯度下降更新模型参数 $\theta$

## 小结 24（Score Function、Score Matching 与随机采样）

设 $p_t(x\mid z),p_t(x)$ 分别表示条件概率路径和边缘概率路径。

那么：

- 条件 score function 是

$$
\nabla \log p_t(x\mid z)
$$

- 边缘 score function 是

$$
\nabla \log p_t(x)
$$

并且，对任意扩散系数 $\sigma_t\ge 0$，下面的 SDE 的轨迹都沿着概率路径演化：

$$
X_0\sim p_{\text{init}},\qquad
dX_t=
\left(
u_t^{\text{target}}(X_t)
+
\frac{\sigma_t^2}{2}\nabla \log p_t(X_t)
\right)dt
+
\sigma_t dW_t
\tag{52}
$$

从而

$$
X_t\sim p_t,\qquad 0\le t\le 1.
\tag{53}
$$

其中 $u_t^{\text{target}}(x)$ 就是前面定义过的边缘向量场。

### Score Matching

为了学习边缘 score $\nabla \log p_t(x)$，我们使用 score network $s_t^\theta$，并通过 denoising score matching 来训练它：

$$
L_{\mathrm{CSM}}(\theta)
=
\mathbb{E}_{z\sim p_{\text{data}},\,t\sim \mathrm{Unif},\,x\sim p_t(\cdot\mid z)}
\bigl[
\|s_t^\theta(x)-\nabla \log p_t(x\mid z)\|^2
\bigr].
\tag{54}
$$

### 高斯概率路径

对于最重要的高斯概率路径

$$
p_t(x\mid z)=\mathcal{N}(x;\alpha_t z,\beta_t^2 I_d),
$$

没有必要分开训练 $s_t^\theta$ 和 $u_t^\theta$，因为它们可以通过下面的公式互相转换：

$$
u_t^\theta(x)=a_t s_t^\theta(x)+b_t x,
\qquad
a_t=\beta_t^2\left(\frac{\dot\alpha_t}{\alpha_t}-\frac{\dot\beta_t}{\beta_t}\right),
\qquad
b_t=\frac{\dot\alpha_t}{\alpha_t}.
$$

训练完成后，我们可以对下面这个 SDE 进行模拟：

$$
X_0\sim p_{\text{init}},\qquad
dX_t=
\left(
u_t^\theta(X_t)+\frac{\sigma_t^2}{2a_t}s_t^\theta(X_t)-\frac{\sigma_t^2 b_t}{2a_t}X_t
\right)dt
+
\sigma_t dW_t
\tag{55}
$$

等价地，也可以写成

$$
X_0\sim p_{\text{init}},\qquad
dX_t=
\left(
\left(a_t+\frac{\sigma_t^2}{2}\right)s_t^\theta(X_t)+b_t X_t
\right)dt
+
\sigma_t dW_t.
\tag{56}
$$

通过模拟这个 SDE，就能得到近似样本

$$
X_1\sim p_{\text{data}}.
$$

而实践中，我们可以经验性地寻找一个最优的 $\sigma_t\ge 0$。
