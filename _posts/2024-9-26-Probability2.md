---
layout: post
title: 概率论:随机变量（绿皮书+额外补充）
date: 2024-9-26 10:00 +0800
tags: [量化绿皮书]
toc: true
---

# 一元随机变量与分布函数

## 离散型随机变量

1. 0-1分布：$P(X=0)=p,P(X=1)=1-p$.
2. 二项分布：X可看做n个0-1分布重复独立试验，试验成功的次数，$P(X=k) = C_n^kp^k(1-p)^{n-k}$，记作$X \sim B(n,p).$
3. 泊松分布：$P(X=k) = \frac{e^{-\lambda}\lambda^k}{k!}$,记作$X \sim P(\lambda)$.
4. 超几何分布：$P(X=k)=\frac{C_a^kC_b^{n-k}}{C_N^n}$，记作$X\sim H(n,a,N)$，就是N个球，a个红球b个白球，不放回取n个，X为红球的个数。
5. 几何分布：$P(X=k)=p(1-p)^{k-1}$，独立重复Bernoulli试验，第k次首次成功的概率。
6. 负二项分布：$P(X=k)=C_{k-1}^{r-1}(1-p)^{kr}p^r$，试验进行到出现r次成功才停止，X为饰演的次数。

可以证明当n足够大，p足够小，且np保持适当大小时，$B(n,p)$可以用$P(np)$近似表述。

## 概率分布函数

概率分布函数定义为$F(x) = P(X\leq x)$，具有以下性质：
1. $P(x_1\leq X \leq x_2) = F(x_2) - F(x_1) + P(X=x_1)$
2. $F(x)$单调不减
3. $0\leq F(x) \leq 1, \lim_{a\rightarrow -\infty}F(a) = 0, \lim_{a\rightarrow \infty}F(a) = 1.$
4. $F(x)$右连续，离散型的$F(x)$没有左连续。

## 连续型随机变量

如果存在一个非负实函数$f(x)$满足$F(x)=\int_{-\infty}^x f(t)dt$，则称X为连续型随机变量，$f(x)$为概率密度函数，具有以下性质：
1. $f(x)\geq 0$
2. $\int_{-\infty}^{\infty}f(t)dt = 1$
3. $\forall x_1<x_2, P(x_1\leq X\leq x_2) = \int_{x_1}^{x_2} f(t)dt$
4. 在$f(x)$的连续点处，$F'(x)=f(x)$
5. $\forall a, P(X=a) = 0.$

常见的连续型随机变量：
1. 均匀分布：$f(x)=\begin{cases}
    \frac{1}{b-a}, x\in(a,b) \\
    0, 其他
    \end{cases}$，记作$X\sim U(a,b).$
    $P(c<X<c+l) = \frac{l}{b-a}, F(x)=\begin{cases}
        0, x<a \\ \frac{x-a}{b-a}, a\leq x < b \\
        1, x\geq b
    \end{cases}$
2. 指数分布：$f(x)= \begin{cases}
    \lambda e^{-\lambda x}, x> 0 \\
    0, x\leq 0
    \end{cases}$，记作$X\sim E(\lambda).$
    $F(x) =\int_{-\infty}^x f(t)dt = \begin{cases}
        1-e^{-\lambda x}, x>0 \\ 0, x\leq 0
    \end{cases}$

    指数分布最重要的性质是无记忆性：$P(X>t_0+t|X>t_0) = \frac{P(X>t_0+t)}{P(X>t_0)} = e^{-\lambda t} = P(X>t)$,即$P(X>t_0+t) = P(X>t_0)P(X>t)$.可以联想到某产品的正常使用时长。
    类似的无记忆性还有几何分布：$P(X-m=n)=P(X=m)P(X=n)$，可以联想到抛硬币。
3. 正态分布：$f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-(x-\mu)^2}{2\sigma^2}}, |x| < +\infty,\sigma>0,|\mu|<+\infty$，记作$X\sim N(\mu,\sigma^2)$服从参数为$(\mu,\sigma)$的正态分布。
    （补充：$\int_{-\infty}^{+\infty}e^{-x^2}dx=\sqrt{\pi}.$）
    正态分布具有以下性质：
    1. $f(x)$关于$x=\mu$对称
    2. $\max f(x) = f(\mu) = \frac{1}{\sqrt{2\pi}\sigma}$
    3. $\lim_{|x-\mu|\rightarrow \infty} f(x) = 0$
    
    当$\mu=0,\sigma=1$的时候称为标准正态分布，密度函数为$\varphi(x)=\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}$,分布函数为$\varPhi(x) = \int_{-\infty}^x \frac{1}{\sqrt{2\pi}}e^{-\frac{t^2}{2}}dt$，具有性质：
    1. $\varPhi(-x) = 1-\varPhi(x)$
    2. 当$X\sim N(\mu,\sigma^2)$时，$\frac{X-\mu}{\sigma} \sim N(0,1) \Rightarrow P(a<x<b) = \varPhi(\frac{b-\mu}{\sigma})-\varPhi(\frac{a-\mu}{\sigma})$

## 一维随机变量函数的分布

若X分布已知，Y=g(X).
1. 当Y是离散型时，找到$\{Y=y_k\}$的等价事件$\{X\in D_k\}$，则有$P(Y=y_k) = P(X\in D_k)$.
2. 当Y是连续型时同样可以求出分布函数$F_Y$，求导获得$f_Y$.

**如果$X$为连续型，密度函数为$f_X(x),\ y=g(X)$，如果$y=g(X)$严格单增(减)，记$y=g(x)$的反函数为$x=h(y)$，则有**
$$f_Y(y) = \begin{cases}
    f_X(h(y))|h'(y)|, y\in D \\
    0, y\notin D
\end{cases}, D为y=g(x)的值域.$$

**1. $X\sim N(\mu,\sigma)$，求$aX+b$的密度函数：**

$y=g(X)=aX+b \Rightarrow x = h(y) = \frac{y-b}{a}, h'(y)=\frac{1}{a} \Rightarrow f_Y(y) = \frac{1}{|a|}f_X(\frac{y-b}{a}).$

**2. $X\sim U(0,\pi),Y=\sin(x)$，求$f_Y(y)$：**
 当$0< y \leq 1, F_Y(y)=P(Y\leq y) = P(X\in[0,\arcsin y]\cup [\pi-\arcsin y,\pi]) = \frac{2\arcsin y}{\pi}$
 因此$f_Y(y)=\begin{cases}
    \frac{2}{\pi} \frac{1}{\sqrt{1-y^2}}, 0 < y \leq 1\\
    0, 其他
 \end{cases}$
（补充：$(\arccos x)' = -\frac{1}{\sqrt{1-x^2}}, (\arctan x)' = \frac{1}{1+x^2}.$）

**有了连续型随机变量后可以去证明$P(AB)=0$并不能证明AB互斥，比如在$(0,1)$上取值，A为(0,0.5],B为[0.5,1).**


# 多元随机变量及其分布

## 二元离散型随机变量

1. 联合分布律：$P(X=x_i,Y=y_j) = p_{ij}, \sum_{i,j}p_{ij}=1, p_{ij}\geq 0.$
2. 边际分布：$p_{i\cdot}=\sum_{j}p_{ij} = P(X=x_i), p_{\cdot j} = \sum_i p_{ij} = P(Y=y_j).$
3. 条件分布：$P(X=x_i | Y=y_j) = \frac{p_{ij}}{p_{\cdot j}}, P(Y=y_j|X=x_i) = \frac{p_{ij}}{p_{i\cdot}}.$

## 分布函数

1. 联合分布函数：$F(x,y) = P(X\leq x, Y \leq y)$，有性质：
   1. $F(x,y)$单调不减
   2. $0\leq F(x,y) \leq 1, F(x,-\infty) = F(-\infty,y) = 0, F(+\infty,+\infty) = 1.$
   3. $F(x,y)$关于x或y右连续
   4. $P(x_1<X\leq x_2, y_1 < Y \leq y_2) = F(x_2,y_2)- F(x_1,y_2) -F(x_2,y_1)+F(x_1,y_1).$
2. 边际分布函数：$F_X(x) = P(X\leq x) = F(x,+\infty), F_Y(y)= F(+\infty,y).
3. 条件分布函数：
   1. 离散型：$F_{Y|X}(y|x_i) = P(Y\leq y| X=x_i)$
   2. 连续型：$F_{Y|X}(y|x) = \lim_{\delta\rightarrow 0^+} P(Y\leq y| x < X \leq x+\delta)$

## 连续型随机变量

1. $F(x,y) =\int_{-\infty}^x\int_{-\infty}^y f(u,v) dudv$，$f(x,y)$称为联合概率密度函数，性质有：
   1. $f(x,y)\geq 0$
   2. $\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty} f(x,y) dxdy = F(+\infty,+\infty) =1$
   3. 在$f(x,y)$连续点上有$\frac{\partial^2F}{\partial x\partial y} = f(x,y).$
   4. $P((X,Y)\in D) = \int\int_D f(x,y) dxdy.$
2. 边际概率密度函数：$F_X(x) = P(X\leq x) = \int_{-\infty}^x[\int_{-\infty}^{+\infty}f(x,y)dy]dx \Rightarrow f_X(x) = \int_{-\infty}^{\infty}f(x,y)dy.$
    同理 $f_Y(y) = \int_{-\infty}^{+\infty}f(x,y)dx.$
3. 条件分布：$F_{Y|X}(y|x) = \lim_{\delta\rightarrow 0^+} P(Y\leq y| x < X \leq x+\delta) = \lim_{\delta\rightarrow 0^+}\frac{F(x+\delta,y)-F(x,y)}{F_X(x+\delta) - F_X(x)} = \int_{-\infty}^y \frac{f(x,v)}{f_X(x)}dv.$
    因此条件概率密度函数$f_{Y|X}(y|x)=\frac{f(x,y)}{f_X(x)}.$
二元均匀分布：$f(x,y) = \begin{cases}
    \frac{1}{S(D)}, (x,y)\in D \\
    0, 其他
\end{cases}$

二元正态分布：$f(x,y) = \frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}}\exp\{-\frac{1}{2(1-\rho^2)}[\frac{(x-\mu_1)^2}{\sigma_1^2} - 2\rho\frac{(x-\mu_1)(y-\mu_2)}{\sigma_1\sigma_2} + \frac{(y-\mu_2)^2}{\sigma_2^2}]\}$，有以下性质：
1. $X\sim N(\mu_1,\sigma_1^2), Y\sim N(\mu_2,\sigma_2^2).$
2. 给定$X=x, Y \sim N(\mu_2+\rho\frac{\sigma_2}{\sigma_1}(x-\mu_1), (1-\rho^2)\sigma_2^2)$
3. 给定$Y=y, X \sim N(\mu_1+\rho\frac{\sigma_1}{\sigma_2}(y-\mu_2), (1-\rho^2)\sigma_1^2)$


**如果X为离散型随机变量，概率分布为p_X(x_i)，由全概率公式有$$P(A)=\sum_iP(A|X=x_i)P(X=x_i)$$**
**如果X为连续型，其密度函数为$p_X(x)$，则$$P(A) = \int_{-\infty}^{+\infty}P(A|X=x)p_X(x).$$**

例子：设$U_1,\cdots,U_n$独立同分布于$(0,1]$上的均匀分布，令$\xi = \min\{n\geq 1: U_1+\cdots+U_n > 1\}$，求$\xi$的概率分布。

令$\xi(x)=\min\{n\geq 1: U_1+\cdots+U_n > x\}$，则有
$P(\xi(x)> 1) = P(U_1\leq x) = x$，进而有递推关系
$$\begin{align*}
    P(\xi(x)>n+1) &= P(U_1+\cdots+U_{n+1} \leq x) \\
    &= \int_{-\infty}^{+\infty}P(U_1+\cdots+U_{n+1} \leq x | U_1 = y)p_{U_1}(y)dy \\
    &= \int_0^1 P(U_2+\cdots+U_{n+1} \leq x-y)dy \\
    &= \int_0^x P(\xi(u)>n)du
\end{align*}$$
由归纳法可以得到$P(\xi(x)>n) = \frac{x^n}{n!}$，因此可以得到$P(\xi>n)=\frac{1}{n!}\Rightarrow P(\xi = n) = \frac{1}{(n-1)!} - \frac{1}{n!}.$

## 随机变量的独立性

离散型：$p_{ij} = p_{i\cdot}p_{\cdot j}, i,j=1,2,\ldots$
连续型：$f(x,y) = f_X(x)f_Y(y), a.e. \Leftrightarrow f(x,y)$几乎处处可以写为$m(x)$与$n(y)$的乘积，**注意f的取值范围，y不能与x有关**

例：二元正态分布中，XY相互独立$\Leftrightarrow \rho = 0.$

## 二元随机变量函数的分布

