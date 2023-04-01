---
layout: post
title: [论文阅读笔记]Sub-linear RACE Sketches for Approximate Kernel Density Estimation on Streaming Data——WWW 2020
subtitle: KDE method
categories: markdown
tags: [论文阅读笔记]
---
<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>
## Sub-linear RACE Sketches for Approximate Kernel Density Estimation on Streaming Data——WWW 2020

### ==Introduction==

传统的KDE需要O(N)kernel function evaluation，o(Nd)memory；

近似估计KDE至$1\pm \epsilon$multiplicative approximation，仍需要无限制访问数据集，有内存限制；

需要一种算法，可以在不存储大量数据情况下，从数据集的一次传递中输出KDE；

- Compress KDE Problem:

- 用于低内存KDE算法的数据流应用：

  【在线数据集摘要】【网络中压缩分类】

### ==Background== LSH和sketch介绍

- LSH：

  <img src="https://raw.githubusercontent.com/yz2022/notes/main/img/202304011924513.png" alt="image-20230401192411484" style="zoom:80%;" />

  LSH family：一组哈希函数，满足在哈希映射下，相似点具有相同哈希值概率大

  哈希函数独立连接：连续使用同一个哈希函数，LSH函数h(·)按顺序应用p次，得到的函数是一个具有碰撞概率$k^p(x, y)$的新哈希函数，新哈希函数范围$[1,R^p]$可以增加两个相似输入映射到相同哈希值的概率(即碰撞)，同时降低不同输入之间碰撞的概率，这使得在大型数据集中更容易有效地识别相似项，这是信息检索和数据挖掘中的常见任务。

  rehashing技巧：LSH函数+universal 哈希函数  组成哈希函数$g(\cdot)$，碰撞概率在hx=hy时一定会碰撞，在hx!=hy时当LSH映射到同一个桶时才会碰撞，即$$\begin{eqnarray}Pr[g(x)=g(y)]&=&Pr(h(x)=h(y))*1+Pr(h(x)!=h(y))*\frac{1}{R}\\&=&k(x,y)+k(x,y)*\frac{1}{R}\end{eqnarray}$$

  

- LSH kernel：由碰撞概率形成的正半定径向核

  > 正半定核是一种基于距离度量的函数，它通过将数据点映射到高维特征空间中，使得在该空间中的点之间的欧几里得距离等于在原始空间中的点之间的距离的函数。在这个特征空间中，正半定径向核可以定义为：K(x, y) = exp(-gamma * ||x - y||^2)
  >
  > 其中，x和y是原始空间中的数据点，||x - y||^2是它们之间的欧几里得距离的平方，gamma是一个超参数，它控制着高斯分布的标准差。当gamma越大时，正半定径向核的峰值会变得更加尖锐，因此模型会更加关注距离中心点较近的数据点。
  >
  > 
  >
  > 而碰撞概率也具有类似的分布，数据点距离越接近，说明碰撞概率越大，所以碰撞概率可以作为LSH kernel

  为了证明$k(x,y)$是半正定核，需要证明对于任意的$n$和$\lbrace x_1,x_2,\dots,x_n\rbrace $，矩阵$K = [k(x_i,x_j)]_{n\times n}$是半正定矩阵。具体地，需要证明：对于任意的向量$c_1,c_2,…,c_n \in R^n$，都有$c^{\top}Kc \geq 0$。

  假设向量$x$和$y$在$n$维欧几里得空间中，它们的距离为$d(x,y)$。由于$f(·)$是单调递减的，因此$d(x,y)$越小，则对应的$k(x,y)$越大。因此我们有以下结论：对于任何$x$和$y$，都有$k(x,y)\geq 0$，当且仅当$x=y$时，$k(x,y)=0$。这意味着对角线元素为非负的。

  下面，我们考虑对于任意的$c_1,c_2,…,c_n \in R^n$，证明$c^{\top}Kc \geq 0$。由于$k(x,y)$是关于$x$和$y$对称的，因此$K$是对称的。可以将$c$展开为$c = [c_1,c_2,\dots,c_n]^{\top}$。因此，$c^{\top}Kc$可以写成：

  $$ c^{\top}Kc = \sum_{i=1}^{n}\sum_{j=1}^{n}c_ic_jk(x_i,x_j) = \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n c_ic_j(k(x_i,x_j)+k(x_j,x_i)) $$

  由于$k(x,y)=k(y,x)$，因此有$k(x_i,x_j)+k(x_j,x_i)=2k(x_i,x_j)$。因此，上述公式可以简化为：

  $$ c^{\top}Kc = \sum_{i=1}^n\sum_{j=1}^n c_ic_jk(x_i,x_j) = \sum_{i=1}^n\sum_{j=1}^n c_ic_jf(d(x_i,x_j)) $$

  由于$f(·)$是单调递减的，因此$d(x_i,x_j)$越小，则对应的$f(d(x_i,x_j))$越大，这意味着$$\sum_{i=1}^n\sum_{j=1}^n c_ic_jf(d(x_i,x_j)) \geq 0$$

  因此，$K$是半正定的。

  

  为了确保文中LSH函数存在LSH kernel，附加LSH属性如下：$k(x, y) ∝ f (d(x, y))$，这个条件充分证明k(x, y)是正半定核。
  文献中的LSH函数大多满足碰撞单调递减性质。

  

- RACE：repeated array-of-counts estimator重复计数数组估计器

  虽然LSH最初是为高维最近邻搜索问题引入的，但该技术最近也被应用于各种函数的自适应采样的无偏统计估计。我们的KDE方法将使用RACE算法，它将LSH视为一种稍微不同的统计估计器。RACE算法将数据集D压缩为L × R个整数计数器的数组a，其中a的每L行都是一个**ACE数据结构（Arrays of (locality-sensitive) count estimators(位置敏感)计数数组估计器）。**
  为了将元素x∈D添加到A中，我们使用一组L独立的LSH函数{h1(x)，…
  hL(x)}。然后，我们增加计数器A[i, hi (x)]，i ={1，…L}。因此，每个计数器记录对应LSH桶的元素数量。

  定理1 ACE estimator
  给定数据集D，具有有限范围[1,R]和参数p的LSH family H，通过连接来自H的p个独立哈希来构造LSH函数<img src="https://raw.githubusercontent.com/yz2022/notes/main/img/202304011925958.png" alt="image-20230401192505923" style="zoom:80%;" />($\mapsto$表示函数映射)
  设A为用h(x)构造的ACE数组，对于任意查询q, ![image-20230401192522183](https://raw.githubusercontent.com/yz2022/notes/main/img/202304011925218.png)

  > 因为![image-20230401192545630](https://raw.githubusercontent.com/yz2022/notes/main/img/202304011925667.png)

  定理2 ACE estimator variance
  给定一个查询q, ACE估计量A [h(q)]的方差服从以下不等式![image-20230401192557295](https://raw.githubusercontent.com/yz2022/notes/main/img/202304011925320.png)

  这些结果表明，重复的ACE或RACE，可以在足够大的重复次数下，以非常低的相对误差估计KDE。

### ==用RACE进行KDE==  将ACE扩展到近似KDE上

对于任何LSH kernel准确报告其KDE

> median-of-means（MoM）：给定一个数据样本，MoM 估计器会打乱数据点，然后将它们分成 k 组，每组 m 个数据点。然后计算每个组的算术平均值。最后，我们计算得到的 k 算术平均值的中值。
>
> ![image-20230401192607940](https://raw.githubusercontent.com/yz2022/notes/main/img/202304011926975.png)
>
> ![](https://raw.githubusercontent.com/yz2022/notes/main/img/202304011926566.png)

某个查询的KDE是数据集中附近元素数量的度量。通过使用几个LSH函数，创建数据集的多个超平面分区，RACE count是归入每个分区的点数，某个查询的RACE count是包含该查询的分区的密度。
因此，每个RACE计数都很好地近似于查询时的KDE。
通过对几个ACE取平均值，我们提高了估算值。
因为ACE估算大幅集中于mean附近，只需要几次重复就就可以提供一个良好的KDE近似。

- Algorithm 1：在线KDE sketch构建和查询

  因为核函数是距离越近碰撞概率越大，所以对于q，其结果对应了更多q附近的元素。因为想估计值=q的概率，所以越接近q概率越高，和核函数性质吻合。

  <img src="https://raw.githubusercontent.com/yz2022/notes/main/img/202304011926355.png" alt="image-20230327201745865" style="zoom:80%;" />

  <img src="https://raw.githubusercontent.com/yz2022/notes/main/img/202304011926903.png" alt="image-20230327204857799" style="zoom:67%;" />

  最终结果应该还会除以$\mathcal{D}$，因为RACE sketch求的是和而不是完整的KDE。

- 实际应用中优点：
  :one:RACE很容易实现，因为A是一个简单的整数计数器数组；

  :two:sketch构造算法完全由哈希函数和数组增量组成；
  :three:虽然在理论上对查询来说，中位数过程是必要的，但实际上，只需报告计数的平均值就足够了；

  :four:由于我们只使用整数数组，我们可以很容易地应用额外的特定于整数的内存优化来进一步压缩A。例如实践中，A通常由许多项组成，这些项要么是零，要么是近似相同的(大)值。我们可以通过以稀疏格式存储A，或者使用各种著名的压缩整数序列(如delta编码)的技术将A打包到压缩的短数组中来节省内存，这种优化对于基于样本的方法是不可能的。
  我们的草图在不增加误差的情况下也是可合并的；
  :five:假设我们使用算法1从D1创建A1，从D2创建A2，使用相同的LSH函数，要构造D1∪D2的估计值，只需要将A1和A2的计数器相加即可。这个属性对于大规模分布式系统是至关重要的，因为A1和A2可以构建在不同的设备上。只要使用相同的LSH函数种子和RACE参数构造两个种族，就可以随时动态地合并或更新它们。
  :six:error-stable合并允许以多种方式并行RACE sketch。首先，L个不同的哈希函数计算可以在绘制和查询过程中并行完成，因为RACE数组的每一行都是完全自包含和独立的。其次，草图过程可以分布在多个process中，每个进程都可以绘制大型数据集的一个子集，并将其更新作为部分RACE草图发送。

### ==Theory==

- RACE for Angular Kernel：

  估计![image-20230327205324540](https://raw.githubusercontent.com/yz2022/notes/main/img/202304011926099.png)

  > 如果$w$是向量，那么$w^\top x$不是表示转置相乘而是表示内积，对应位置相乘相加；同时，矩阵卷积对应的每一个点相当于卷积核与原矩阵做内积。
  >
  > $w^\top x=内积=||w||\cdot ||x||\cdot cos\theta$，所以$h_w(x)结果正负取决于$$w$和$x$的方向（夹角）。

  碰撞概率形成半正定核

  ![image-20230328114559350](https://raw.githubusercontent.com/yz2022/notes/main/img/202304011926636.png)**如何计算的？**

  首先，$k (x, y)∝f (d (x, y))$，而$d (x, y))$可用$\theta(x,y)$替代，所以$k$是一个关于$\theta$的单调递减函数。取极值，1)$\theta=0,k=1$ ;2)$\theta=\pi,k=0$ ;3)$\theta=\pi/2,k=1/2$，在笛卡尔坐标系作图并待定系数得k(x,y)=…

- 所有LSH kernel的估计：

  许多LSH kernel不具有有限范围R，所以碰撞概率不在有限范围内，相应的LSH kernel难以估计。将其通过rehashing技术使得索引控制在R范围内

  > 误差函数erf：![image-20230328150019194](https://raw.githubusercontent.com/yz2022/notes/main/img/202304011926825.png)
  >
  > ![image-20230328150232058](https://raw.githubusercontent.com/yz2022/notes/main/img/202304011926434.png)
  >
  > 指数kernel： 指数核函数就是高斯核函数的变种，它仅仅是将向量之间的L2距离调整为L1距离，这样改动会对参数的依赖性降低，但是适用范围相对狭窄。其数学形式如下：
  >
  > <img src="https://raw.githubusercontent.com/yz2022/notes/main/img/202304011926553.jpeg" alt="img" style="zoom:67%;" />
  >
  > 欧氏距离的LSH function：
  >
  > ![image-20230328205727027](https://raw.githubusercontent.com/yz2022/notes/main/img/202304011926032.png)



- 任意kernel的估计：kernel从$k(x,q)$变化为$k(d(x,q))$

  当LSH kernel的形式不是$k(x,q)$而是$k(d(x,q))$时，碰撞概率不再是LSH kernel。

  >  ![image-20230328153504869](https://raw.githubusercontent.com/yz2022/notes/main/img/202304011927878.png)
  >
  >  $碰撞概率z(x,q)=f(d(x,q))可用RACE估计碰撞概率z的KDE \\ \to d(x,q)=f^{-1}(f(d((x,q))))\\ \to kernel=k(f^{-1}(f(d((x,q))))) \\ \to g(z)=k(f^{-1}(z(x,q)))$
  >
  >  [20]中的泰勒级数系数：![image-20230328163532830](https://raw.githubusercontent.com/yz2022/notes/main/img/202304011927488.png)用于求逆函数分解成多项式的结果，
  >
  >  进行KDE估计例子：
  >
  >  1）p=1：估计$Z=\sum_{x\in \mathcal{D}}z(x,p)^1$，$g(z)=k[C_1z^1+C_2z^2+…]$，因为最终要求的是$g(z)$的KDE，所以$g(z)$的KDE和$Z$满足$kC_1$函数关系，将Z带入$g(z)$即可得到第一项，作为KDE的第一项；

  用泰勒级数逆方法从$k(x,q)$估计$d(x,q)$：

  > 碰撞概率$z$就是$k(x,q)$，已知$k(x,q)$表达式，又$k(x,q)=z(x,q)$，由于kernel $d(x,q)=f^{-1}(z(x,q))$，对$d(x,q)$进行泰勒展开，可得各项系数，加上已知的z值即可得到d

### ==Discussion==

- Computation 和Memory的权衡：

  从内存来看，R=3取最佳；从计算复杂度来看，更新和查询RACE sketch的时间复杂度都为O(N)，导致内存和计算的权衡；

  可以选择以下三个参数中的任意两个:error、内存和更新成本。
  我们可以用一个模拟实验来理解这种权衡。
  首先生成一个包含50万个点的聚类数据集，用不同的参数组构建了多个RACE sketch。
  我们发现R = 3, L = 10×103的RACE有0.5%的相对误差。
  同样的误差可以通过R = 4 × 103和L = 2 × 103的RACE草图实现。
  第一个草图比第二个草图小260倍，但每次更新需要5倍的hash计算。
  从这个小实验中，我们看到极端压缩是可能的，但会增加计算成本

  幸运的是，哈希计算并不是非常昂贵。
  在实践中，L通常可以小于200以获得良好的结果(ε < 5%)。
  LSH计算并行化很简单，可以在硬件[28]中实现。
  许多感兴趣的LSH函数只需要内积计算，因此非常适合在GPU上实现。
  此外，如果我们使用**稀疏数组**来实现RACE，我们可以有效地获得无限的重散列范围，而不会显著增加内存。
  由于许多RACE计数通常为零，这种方法可以同时减少内存和运行时。

  > 稀疏数组：
  >
  > <img src="https://raw.githubusercontent.com/yz2022/notes/main/img/202304011927772.png" alt="image-20230328173640073" style="zoom:80%;" />

- Privacy：

  RACE具有独特的隐私保护特性。
  我们不存储任何元素或数据属性，而是将数据集流到RACE草图中，而不存储在任何地方。
  内核合并方法存储数据集的聚合摘要，但仍然需要用户数据属性的平均值。
  RACE是通过一个随机哈希函数构造的，它根本不需要存储或合并元素。

### ==Experiments==

使用p-stable Euclidean LSH kernel

估计angular kernel density

- baseline：随机抽样；稀疏kernel近似；基于哈希的sketch

  1）RS 随机抽样：数据集中选择一组点，同等加权

  2）HBS 基于哈希的sketch：用一个LSH函数将数据集中每个点hash到一个哈希表中，从哈希表中采样点，根据每个哈希桶大小对他们进行加权

  > 哈希表：![img](https://raw.githubusercontent.com/yz2022/notes/main/img/202304011927788.png)
  >
  > 哈希桶：![image-20230328181105447](https://raw.githubusercontent.com/yz2022/notes/main/img/202304011927364.png)

  3）稀疏kernel近似（SKA）：找到数据集中的k个中心位置并加权

  ![image-20230328213245875](https://raw.githubusercontent.com/yz2022/notes/main/img/202304011927435.png)

  
