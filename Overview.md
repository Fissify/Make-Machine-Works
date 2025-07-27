## 一、总论

### （一）简单介绍

#### 1.机器学习与因果推断

机器学习的指向是预测，即以历史数据为基础，通过X去预测Y。（***泛化**是机器学习的重要概念，指调试的算法具有适用性，能够对未知数据进行预测*）

因果推断的指向是关系，通过构建因果关系验证假设。

#### 2.机器学习关键术语

|      因果推断      |  机器学习  |
| :----------------: | :--------: |
|  自变量、解释变量  | 表征、特征 |
| 因变量、被解释变量 |    响应    |
|       观测值       |    案例    |
|        模型        |    算法    |

#### 3.有监学习与无监学习

**有监督学习**是根据已有数据集，知道输入和输出结果之间的关系，并根据这种关系训练得到一个最优模型（*既有特征也有标签*）。

**无监督学习**处理的是只有输入变量，没有相应的输出变量的训练数据。

### （二）基本原理

#### 1.过拟合和欠拟合

**过拟合**是指，算法在训练样本中过于优越，导致在测试数据集中表现不佳。

<div style="text-align: center;">
  <img src="https://pic.baike.soso.com/ugc/baikepic2/0/20220319102224-1953512571_jpeg_1200_800_89580.jpg/800" width="300" height="auto" />
  <span style="display:block; margin-top: 10px; font-family: 'STHeiti', '华文细黑', sans-serif;">狸花猫</span>
</div

<div style="text-align: center;">
  <div style="display: flex; justify-content: center; gap: 10px;">
    <img src="https://pic.baike.soso.com/ugc/baikepic2/0/ori-20200801233300-366433400_jpeg_2000_1123_152372.jpg/800" width="300" height="auto" />
    <img src="https://pic.baike.soso.com/ugc/baikepic2/0/20230628120655-363553596_jpg_2000_1424_426735.jpg/800" width="300" height="auto" />
     </div>
  <span style="display:block; margin-top: 10px; font-family: 'STHeiti', '华文细黑', sans-serif;">橘猫与柯基犬</span>
</div>

> 例如，根据特征识别出猫和狗，但训练集中都是狸花猫的数据，因而训练集中的算法表现十分优秀。
> 但当测试集中不再是狸花猫的数据而是橘猫时，算法准确度就会变得很糟糕，难以分辨这是猫还是柯基犬。

**欠拟合**是指，算法在训练集和测试集中的表现都不好（*包含了太多无意义的噪音*）。

#### 2.偏差和方差

由于机器学习的重点是预测。因此，判断算法好坏即估计值与真实值之间的差距，在机器学习中称为**偏差（*Bias*）**，其定义为：
$$
Bias=\mathbb{E}\hat f(x)-f(x)
$$
其中$\hat{f(x)}$为给定$x$的估计值，$f(x)$ 为真实值。

**方差**是衡量在大量抽样过程中，估计量$\hat{f(x)}$本身围绕着其期望$\mathbb{E}\hat f(x)$的波动幅度，其定义为：
$$
Var=\mathbb{E}[\hat f(x)-\mathbb{E}\hat f(x)]^2
$$
事实上，算法的预测能力取决于**均方误差(*MSE*)**：
$$
\begin{align}
MSE(\hat{f}(x)) &= \mathbb{E}[(y - \hat{f}(x)]^2= \mathbb{E}[(f(x) + \epsilon - \hat{f}(x)]^2 \\
											&= \mathbb{E}[f(x)- \mathbb{E}\hat f(x)+\mathbb{E}\hat f(x)-\hat f(x)+\epsilon]^2 \\
											&=[\mathbb{E}\hat f(x)-f(x)]^2+\mathbb{E}[\hat f(x)-\mathbb{E}\hat f(x)]^2+\mathbb{E}(\epsilon^2)\\
											&=Bias^2+Variance+Var(\epsilon)
\end{align}
$$
也就是说，算法的预测性能同时受偏差和方差影响。一般来说，偏差小而方差大，会出现“过拟合”；偏差大而方差小则是“欠拟合”。

<div style="text-align: center;">
<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6PhibbxpWaH0jG1EbAib9icsFTkpEAia9KQeq5MZUvXm9F0I8Y5sMZnFX2DR0GYDRrjsWmBQkQfGMMz4KAw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" width="300" height="auto" />
   <img src="https://mmbiz.qpic.cn/mmbiz_jpg/YpS8ucOiaGvMWBvUav04GG4ULmatseBfXA6fCHM104AKzlysiaiccJhPtqNq3DJXJMbgjAdrElhATut5wjS5pmyhg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" width="300" height="auto" />
     </div>
  <span style="display:block; margin-top: 10px; font-family: 'STHeiti', '华文细黑', sans-serif;">偏差和方差</span>
</div

总的来说，偏差表示算法是否准确，方差表示算法是否稳定。

#### 4.泛化

泛化是机器学习的首要目标，甚至是唯一目标。由于偏差和方差之间是此消彼长的关系，因此如何实现二者之间的最优是个难题。在这过程中，机器学习有三个最重要的方式以实现目标。

**（1）正则化**

在我们进行预测中，会用到最小二乘法。基于高斯-马尔可夫定理，我们很容易得到无偏估计，但无偏估计并不一定等于最优估计量。以最小二乘法建模时，我们的偏差为0，方差为$\sigma^2$。当模型过于复杂，$\sigma^2$会变得非常大，此时模型处于过拟合状态，泛化性能很低。因此需要通过引入惩罚项牺牲无偏性以减小方差，这个思路就叫做正则化。正则一词的本意为“*使某物变得规则、标准或规范*”，正则化就是使过拟合的模型变得更简单、规范，以避免模型在训练数据上学习到不必要的复杂性或噪声。

其中，最常用的正则化方式为岭回归和Lasso回归。具体来说，岭回归和Lasso回归都是在OLS的目标函数基础上加上惩罚项：
$$
\begin{align}
\text{岭回归}&:min(\sum_{i}(y_i-X_i^T\beta)^2+\lambda \sum_{i=1}^p\beta_j^2)\\
\text{Lasso回归}&:min(\sum_{i}(y_i-X_i^T\beta)^2+\lambda \sum_{i=1}^p|\beta_i|)
\end{align}
$$
惩罚项是对过拟合模型的一种“惩罚”，目的是限制模型的自由度。试想一下，你在一个城市的路上开车，目标是找到一个最佳路径，但你不能在每个街道上都停下来（*过拟合*），你希望找到一个尽可能简短、平稳的路径来避免走冗余的弯路。惩罚项就像是在路径的每个转弯点施加了一个“罚款”，每次转弯越多（*模型越复杂*）就要支付更多的“罚款”，最终你会选择一个更加简单、平稳的路径。

**（2）集成算法**

简单来说，集成算法的核心就是通过一些方式提高算法的预测性能（*力大砖飞*），例如，进行多次有放回的抽样求平均值。

**（3）深度学习法**

以神经网络为例，进行多次嵌套。

### （三）特点与应用

#### 1.特点

机器学习有以下几种优势与特点：

处理高维数据、处理非结构化数据、处理非线性关系、强调模型的泛化性能、对异质性估计重视、数据驱动。

#### 2.应用

机器学习的主要应用范围：

预测、生成数据（*将海量的非结构化数据处理成社会科学能够使用的结构数据*）、因果识别。
