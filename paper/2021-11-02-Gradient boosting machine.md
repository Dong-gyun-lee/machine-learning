---
title: Gradient Boosting machines, a tutorial
toc: true
author_profile: false
use_math: True
---


# Gradient Boosting machines, a tutorial

출처: https://www.frontiersin.org/articles/10.3389/fnbot.2013.00021/full

## 1. Introduction  

앙상블 방법은 weak learner 들을 결합하여 더 강력한 앙상블 예측을 얻으려고 하는 방법이다.
특히 앙상블 방법 중 부스팅 방법의 아이디어는 앙상블에 새로운 모델을 추가하는 것이다. 

<br>

## 2. Methodology
### 2.1 Function estimation
<br>

$$
\begin{equation}
\begin{aligned}
\hat{f}(x) & = y \\
\hat{f}(x) = & \,\underset{f(x)}{\operatorname{argmin}}\,{\psi(y,f(x))}
\end{aligned}
\end{equation}
$$

- (1)번 식을 보면 loss function ${\psi(y,f(x))}$ 을 가장 작게 하는 f 함수를 찾는 것이 목적이라고 볼 수 있다.

<br>

$$
\begin{equation}
\begin{aligned}
\hat{f}(x) = & \,\underset{f(x)}{\operatorname{argmin}}\,\,\,{\underbrace{E_x[{\overbrace{E_y(\psi[y,f(x)])}^{\text{expected y loss}}}\,|x]}_{\text{expectation over the whole dataset}}}
\end{aligned}
\end{equation}
$$

- 반응변수의 분포에 따라 ${\psi(y,f(x))}$ 의 형태를 고려해 볼 수 있다. 예를 들어 반응변수가 이항 분포이면 이항 손실 함수(binomal loss function) 를 고려해 볼 수 있다.

<br>

$$
\begin{equation}
\begin{aligned}
\hat{f}(x) = f(x,\hat{\theta})
\end{aligned}
\end{equation}
$$

$$
\begin{equation}
\begin{aligned}
\hat{f}(x) = \, \underset{\theta}{\operatorname{argmin}}\,E_x[E_y(\psi[y,f(x,\theta)])\,|x]
\end{aligned}
\end{equation}
$$

- 함수 추정 문제를 다루기 쉽게 만들기 위해 논문에서는 함수 공간을 함수 $f(x,\theta)$ 의  Parametic family 의 형태로 제한하여서 함수 최적화 문제를 위와 같이 변경하였다. <br> 아무래도 non-parametic 한 함수에서 loss function을 가장 작게 만들어 주는 $f(x)$ 를 찾아주는 것보다는 함수의 모수를 찾아주는 방법이 쉬운 방법이므로 합리적인 변환이라고 생각한다.

<br>

### 2.2 Numerical optimization
<br>
- 함수의 추정은 Numerical method를 이용하여 추정한다.

<br>

$$
\begin{equation}
\begin{aligned}
\hat{\theta} = \sum_{i=1}^{M}\hat{\theta}_i
\end{aligned}
\end{equation}
$$

- M 단계의 반복에서 모수 추정치는 (5)번식과 같이 증분 형식(incremental form)으로 표현 가능하다.

<br>

$$
\begin{equation}
\begin{aligned}
J(\theta) = \sum_{i=1}^{N}{\psi(y_i,f(x_i,\hat{\theta}))}
\end{aligned}
\end{equation}
$$

- Paramter estimation 에서 가장 간단한 방법은 steepest gradient discent 이다. 주어진 N개의 데이터 포인트들에서 우리는 emprical loss를 줄여주는 방향으로 단계를 진행해주고자 한다.
- 즉, 최적화 절차는 $\nabla J(\theta)$ 의 방향으로 계속해서 걔선해나간다.

<br>

- Steepest descent procedure

<br>

1. 초기값 $\hat{\theta}$_0 을 정한다. 그리고 각 interation t을 반복:
2. 이전의 iteration 으로 부터 $\hat{\theta}^t$ 추정:

$$
\begin{equation}
\begin{aligned}
\hat{\theta}^t = \sum_{i=0}^{t-1}\hat{\theta}_i
\end{aligned}
\end{equation}
$$

3. 주어진 파라미터들로 $\nabla J(\theta)$ 계산:

$$
\begin{equation}
\begin{aligned}
\nabla J(\theta) = {\nabla J(\theta_i)} = \bigg[{\partial J(\theta)\over\partial J(\theta_i)}\bigg]_{\theta=\hat{\theta}^t}
\end{aligned}
\end{equation}
$$

4. $\hat{\theta}_t$ 계산:

$$
\begin{equation}
\begin{aligned}
\hat{\theta_t} \leftarrow -\nabla J(\theta)
\end{aligned}
\end{equation}
$$

5. 새로운 추정치 $\hat{\theta}_t$ 를 앙상블에 추가

<br>

### 2.3 Optimization in function space

$$
\begin{equation}
\begin{aligned}
\hat{f}(x) = \hat{f}^M(x) = \sum_{i=0}^{M}\hat{f_i}(x)
\end{aligned}
\end{equation}
$$

$M$: 반복 수 <br>

$\hat{f}_0$ : 함수의 초기 추정치 <br>

${\hat{\{f\}}}_{i=1}^{M}$ : 함수 증분(= boost)

<br>

- 부스팅 방법과 기존의 머신러닝 방법들의 차이는 최적화(optimization)가 함수 공간(functional space)
에서 유지된다는 것. 즉, 추정된 함수 $\hat{f}$ 를 위의 (13)식처럼 가법 함수 형태(additive functional form)으로 매개 변수화할 수 있다.


$$
\begin{equation}
\begin{aligned}
\hat{f_t} \leftarrow \hat{f}_{t-1} + {\rho}_th(x,\theta_t)
\end{aligned}
\end{equation}
$$

$$
\begin{equation}
\begin{aligned}
(\rho_t,\theta_t) =\, \underset{\rho, \,\theta}{\operatorname{argmin}}\,\sum_{i=1}^{N}\,\psi(y_i,\hat{f}_{t-1})+\rho h(x_i,\theta)
\end{aligned}
\end{equation}
$$

- functional approach를 실제로 실현하기 위해 함수군을 매개변수화하는 유사한 전략을 사용한다. <br>  여기서는 전체 앙상블 함수 $f(x)$ 와 구별하기 위해 'base-leaner' 함수 $h(x,\theta)$를 사용한다. <br> 그로 인해, optimizaion rule (11),(12) 식과 같이 다시 정의 된다. 결국은 어떠한 loss function을 최소화하는 'base-learner'의 모수 $\theta$ 와 $\rho$ 를 찾는 것이라고 볼 수 있다.

<br>

### 2.4 Gradient boost algorithm

$$
\begin{equation}
\begin{aligned}
g_t(x) = E_y\bigg[{\partial\psi(y,f(x))\over\partial f(x)}|x\bigg]_{f(x)=\hat{f}^{t-1}(x)}
\end{aligned}
\end{equation}
$$

- 임의로 loss function과 base-learner 모델을 특정화할 수 있다. 하지만, 그런 경우에 실제로 모수 추정치를 얻어내기가 힘들다. 따라서, negative gradient 방향으로 가장 평행한 새로운 함수 $h(x,\theta_t)$ 를 찾는 것이 제안 되었다. 여기서 왜 negative gradient 방향이 loss function을 최소화시키는 방향인가에 대해서 생각해볼 때, 벡터의 관점에서 생각해보면 negative gradient 방향이 왜 오차를 줄여주는 방향인지 쉽게 알 수 있는 것 같다.

<br>

$$
\begin{equation}
\begin{aligned}
(\rho_t,\theta_t) = \underset{\rho, \,\theta}{\operatorname{argmin}}\,\sum_{i=1}^{N}\,[-g_t(x_i)+\rho h(x_i,\theta)]^2
\end{aligned}
\end{equation}
$$

- 일반적인 솔루션을 찾는 대신 $g_t(x)$ 와 가장 상관관계가 높은 새 함수 증분(function incredment)를 찾는 것으로 바꾸어 생각할 수 있다. 이를 통해서 매우 어려운 최적화 작업을 클래식한 최소-제곱 최적화 작업으로 대체할 수 있다.
- 위의 과정들을 쉽게 생각해서 negative gradient, 즉 오차를 줄여주는 방향을 base-learner가 학습한다고 생각할 수 있다. 그러므로 base-learner $h(x,\theta)$ 는 gradient를 학습한다 라고 볼 수도 있을 것 같다.

<br>

**Algorithm:** <br>
1. initiallize $\hat{f}_0$ with a constant
2. **for** $\,t=1\,$ to $M$ **do**
3. $\quad$ compute the negative gradient $g_t(x)$
4. $\quad$ fit a new base-learner function h(x,$\,\theta_t$)
5. $\quad$ find the best gradient descent step-size $\rho_t$:<br>
$\quad$ $\rho_t = \, \underset{\rho}{\operatorname{argmin}}\,\sum_{i=1}^{N}\,\psi(y_i,\hat{f}_{t-1}(x_i))+\rho_t h(x_i,\theta_t)$
6. $\quad$ update the function estimate: <br>
$\quad$ $\hat{f_t} \leftarrow \hat{f}_{t-1} + {\rho}_th(x,\theta_t)$
7. $\quad$ **end for**

> 
<br>

## 3. GBM DESIGN
### 3.1 Loss-function families

-
<br>

학습 임무에 따라 다양한 loss function 을 고려할 수 있다. <br>
변수가 연속형 변수, 범주형 변수냐에 따라서 다양한 loss-function을 고려해볼 수 있다. 이 부분은 다른 모델들에서 loss-function을 고려하는 것과 비슷하고 논문을 참고하면 쉽게 볼 수 있으므로 이번 리뷰에서는 생략한다. <br>

<br>
### 3.2 base-learner

base-learner도 Linear model, Smooth model, Decision Tree 등 다양한 함수를 고려할 수 있다. 여러 클래스의 base-learner model을 하나의 GBM에서 동시에 고려하는 것도 가능하다. 또한 변수 간의 상호작용 term 도 base-learner를 통해 학습이 가능하다.

<br>

#### 3.2.1 Additive base-learners
<br>

Additive base-learner model은 설명 변수들 사이에 교호작용(interaction) 이 없다고 가정한다. Additive GBM 모델은 앞에서 설명한 알고리즘과 다르게 각 반 복에서 무작위로 선택된 일부 변수 위에 구축된 여러 추가 base-learner의 후보들이 동시에 적합된다. 그런 다음 잔차 제곱합 기준에 따라서 최적의 모형이 선택된다. 이 과정에서는 많은 설명 변수가 생략되는 상황이 종종 발생하며, 자연스럽게 'Sparse solution을 제공한다
<br>

#### 3.2.2 Decision tree base-learners

<br>

Decision tree base-learner model은 변수 간의 교호작용을 고려해줄 수 있다. 우리가 알고 있는 tree model의 방법과 같다.

## 4. Regularization
Overfitting 을 방지하기 위한 Gradient boosting 의 여러가지 방법들을 여기서 소개하고 있다.

<br>

### 4.1 Subsampling

<br>
모델을 적합하는 과정에서 train data의 일부만 랜덤하게 선택해서 base-learner를 학습시키는 것.
데이터가 충분히 크다면 bag=0.5 값이 경험적으로 좋은 결과를 얻을 수 있게 해주는 값이라고 한다.

<br>

### 4.2 Shrinkage

<br>

모델 복잡도를 제어할 때 shrinkage 를 이용한다. 간단하게 설명하자면 learning rate와 같은 효과라고 생각하면 된다. 아래의 식에서 $\lambda$ 가 shirinkage를 나타낸다.

$\qquad\qquad\qquad\qquad\qquad \hat{f_t} \leftarrow \hat{f}_{t-1} + \lambda{\rho}_th(x,\theta_t)$

<br>

### 4.3 Early stopping


<br>

shrinkage 모수 $\lambda$ 에 따라 반복횟수 M이 달라지고, Early stopping을 통해서 최적의 M을 얻을 수 있다.


<br>

## 5. Model Interpretation
---------------------------------------
### 5.1 RELATIVE VARIABLE INFLUENCE

<br>

$Influence_j(T) = \sum_{i=1}^{L-1}\,I_i^21(S_i=j)$

- 단일 트리 $T$에서 변수 $j$ 의 영향력 <br> $L$ 은 split 수, $S_i$ 는 current splitting variable, $I_i^2$ 은 empiriacal squared improvement

$Influence_j = \sum_{i=1}^{M}\,Influence_j(T)$
- 앙상블 모형 전체에서 변수 $j$의 영향력