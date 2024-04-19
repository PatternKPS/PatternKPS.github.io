---
layout:     post
title:      "Singular Value Decomposition (SVD)"
date:       2024-03-15 15:58:00
author:     "Kristina"
header-mask: 0.3
mathjax:      true
tags:
    - svd
---

### Prelimenary


Let bold-face lower-case letters (like $\mathbf{a}$) refer to <font color='yellow'>vectors</font>, bold-face capital letters $\mathbf{A}$ refer to <font color='yellow'>matrices</font>, and italic lower-case letters (like $\mathcal{a}$) refer to <font color='yellow'>scalars</font>. Think $\mathbf{A}$ is a  square or rectangular. Its rank is $p$. We will diagonalize this $A$, but not by $X^{-1} AX$. The *eigenvectors* in $X$ have three big problems:
1. They are usually not orthogonal
2. There are not always enough *eigenvectors*, and
3. $Ax = \lambda x$ requires $A$ to be a square matrix

The **<font color='yellow'>singular vectors</font>** of $A$ solve all those three problems in a perfect way.  

### Defenition

1. The SVD of a matrix is a sort of change of coordinates that makes the matrix simple, a *generalization of diagonalization*. It has some interesting algebraic properties and conveys important geometrical and theoritical insights about **linear transformations**.

2. The **singular value decomposition (SVD)** of a matrix is a **factorization** of that matrix **into three matrices**, where the factorization has the form $\mathbf{U\Sigma V}$. 

3. A **<font color='yellow'>factorization</font>** is called the **<font color='yellow'>eigendecomposition</font>** of A, also called
the **<font color='yellow'>spectral decomposition</font>** of $\mathbf{A}$.

4. $\mathbf{U}$ is an $m \times p$ matrix, $\mathbf{\Sigma}$ is a $p\times p$ diagonal matrix, and $\mathbf{V}$ is an $n \times p$ matrix, with $\mathbf{V}^T$ being the transpose of $V$, a $p \times n$ matrix. The value $p$ is called the **<font color='green'>rank</font>**. 

5. The diagonal entries of $\mathbf{\Sigma}$ are referred to as _the singular values of_ $\mathbf{A}$. The columns of $\mathbf{U}$ are typically called *the left-singular vectors* of $\mathbf{A}$, and the columns of $V$ are called *the right-singular vectors of* $\mathbf{A}$. 

6. Unfortunately not all matrices can be diagonalized. **<font color='green'>SVD is a way to do something like diagonalization for any matrix, even non-square matrices</font>**.

### Application

1. In many applications, the data matrix $A$ is close to a matrix of low rank and it is useful to find a low rank matrix which is a good approximation to the data matrix . We will show that from the singular value decomposition of $A$, we can **<font color='green'>get the matrix</font>** $B$ **<font color='green'>of rank</font>** $p$ **<font color='green'>which best approximates </font>**$A$.

2. Principal component analysis (PCA)

3. Clustering a mixture of spherical Gaussians

4. To solve discrete optimization problem

5. For spectral decomposition

6. It can be used to **<font color='green'>reduce the dimensionality </font>**, i.e., the number of columns, of a dataset. 

7. It is often used in **<font color='green'>digital signal processing</font>** for **<font color='magenta'>noise reduction</font>**, **<font color='magenta'>image compression</font>**, and other areas.


### Mathematics behind SVD

**<font color='green'>Theorem 1.</font>** For any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, there exist two orthogonal matrices $\mathbf{U} \in \mathbb{R}^{m \times m}$, $\mathbf{V} \in \mathbb {R}^{n \times n}$, and a non-negative "diagonal" matrix $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$ (of the same size as $\mathbf{A}$). Such that the SVD of $m \times n$ matrix $\mathbf{A}$ is given by the formula:

$$
\mathbf{A}_{m\times n} = \mathbf{U}_{m \times m} \mathbf{\Sigma}_{m \times n} \mathbf{V}_{n \times n}^T
$$


where:
- $\mathbf{U}$: $m \times m$ matrix of the orthonormal _eigenvectors_ of $\mathbf{AA}^T$.
- $\mathbf{\Sigma}$: an $m \times n$ matrix whose $i^{th}$ diagonal entry equals the $i^{th}$ singular value $\sigma_i$ for $i=1, 2, \ldots, p$. All other entries of $\mathbf{\Sigma}$ are zero.
- $\mathbf{V}^T$: transpose of a $n \times n$ matrix containing the orthonormal eigenvectors of $\mathbf{A}^T \mathbf{A}$.


> $ \boxed{\color{yellow}\mathbf{U} \Longleftrightarrow  \mathbf{AA}^T = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T \cdot \mathbf{V} \mathbf{\Sigma}^T \mathbf{U}^T= \mathbf{U} (\mathbf{\Sigma} \mathbf{\Sigma}^T) \mathbf{U}^T} $ and 
> $ \boxed{\color{yellow}\mathbf{V} \Longleftrightarrow  \mathbf{A}^T\mathbf{A} =  \mathbf{V} \mathbf{\Sigma}^T \mathbf{U}^T \cdot \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T = \mathbf{V} (\mathbf{\Sigma}^T \mathbf{\Sigma} ) \mathbf{V}^T} $


**<font color='green'>Proof:</font>** Given any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, the SVD can be though of as solving a matrix equation for three unknown matrices (each with certain constraint):

\begin{equation*}
\color{yellow}\mathbf{A}=\underbrace{\mathbf{U}}_{\text{orthogonal}} \cdot \underbrace{\mathbf{\Sigma}}_{\text{diagonal}} \cdot \underbrace{\mathbf{V}^T}_{\text{orthogonal}}
\tag{2}
\end{equation*}

Suppose such solutions exist.

- Knowing: 
$$\mathbf{A}^T\mathbf{A}=\mathbf{V} (\mathbf{\Sigma}^T \mathbf{\Sigma} ) \mathbf{V}^T$$
This tells us how to find $\mathbf{V}$ and $\mathbf{\Sigma}$ (which contain the eigenvectors and square roots of eigenvalues of $\mathbf{A}^T\mathbf{A}$, respectively).

- After we have found both $\mathbf{V}$ and $\mathbf{\Sigma}$, rewrite the matrix equation as 
$$
\mathbf{AV} = \mathbf{U\Sigma}
$$
or in columns,
$$
\mathbf{A} \begin{bmatrix} \mathbf{v}_1 \ldots & \mathbf{v}_p ~ \mathbf{v}_{p+1} \ldots & \mathbf{v}_n\end{bmatrix} = \begin{bmatrix} \mathbf{u}_1 \ldots & \mathbf{u}_p ~ \mathbf{u}_{p+1} \ldots & \mathbf{u}_m\end{bmatrix} ~ \begin{bmatrix} \mathbf{\sigma}_1  &  & & &\\  & \ddots & & &\\ & & \sigma_p & & & \\ & & & & & \\ & & & & & \\  \end{bmatrix}
$$

By comparing columns, we obtain
$$
\mathbf{Av}_i=
\begin{cases}
\sigma_i \mathbf{u}_i,      &1<i \leq p ~~~~~~ \text{(non-zero singular values)}\\ 
0,      & p<i \leq n
\end{cases} 
$$


This tells us how to find the matrix $\mathbf{U} : \mathbf{u}_i = \frac{1}{\sigma_i} \mathbf{A} \mathbf{v}_i$ for $1\leq i \leq p$.

Let $\mathbf{B} = \mathbf{A}^T \mathbf{A} \in \mathbb{R}^{n \times n}$. Then $\mathbf{B}$ is square, symmetric, and positive semidefinite. 

Therefore, by the <font color='yellow'>_Spectral Theorem_</font>, $\mathbf{B} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$ for an orthogonal $\mathbf{V} \in \mathbb{R}^{n \times n}$ and diagonal $\mathbf{\Lambda} = \text{diag} (\lambda_1, \ldots, \lambda_n)$  with $\lambda_1 \geq \ldots \geq \lambda_p > 0 = \lambda_{p+1}=\ldots = \lambda_n$ (where $p = \text{rank} (\mathbf{A})\leq n)$.

Now let $\sigma_i = \sqrt{\lambda_i}$ and correspondingly from the matrix $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$:

$$\color{yellow} \mathbf{\Sigma} = \begin{bmatrix} \text{diag} ( \sigma_1, \ldots, \sigma_p) & \mathbf{O}_{p \times (n-p)} \\ \mathbf{O}_{(m-p) \times p} & \mathbf{O}_{(m-p)\times (n - p)} \end{bmatrix}$$

Define also

$$ \mathbf{u}_i = \frac{1}{\sigma_i} \mathbf{Av}_i \in \mathbb{R}^m, ~~~~~~~~~~~ \text{for each} ~~ 1\leq i\leq p$$


Then $\mathbf{u}_1, \ldots, \mathbf{u}_p$ are orthonormal vectors. To see this,

$$
\begin{align}
\mathbf{u}_i^T \mathbf{u}_j &= \bigg ( \frac{1}{\sigma_i} \mathbf{Av}_i \bigg)^T ~ \bigg ( \frac{1}{\sigma_j} \mathbf{Av}_j \bigg) & = \frac{1}{\sigma_i \sigma_j} \mathbf{v}_i^T ~ \underbrace{\mathbf{A}^T \mathbf{A}}_{\mathbf{B}} ~ \mathbf{v}_j &\\
 & = \frac{1}{\sigma_i \sigma_j} \mathbf{v}_i^T (\lambda_j \mathbf{v}_j) = \frac{\sigma_j}{\sigma_I} \mathbf{v}_i^T \mathbf{v}_j & &(\lambda_j = \sigma_j^2)\\
 & = \begin{cases}
1,      & i = j\\
0,      & i \neq j
\end{cases}
\end{align}
$$

Choose $\mathbf{u}_{p+1}, \ldots, \mathbf{u}_m \in \mathbb{R}^{m}$ (through basis completion) such that

$$
\mathbf{U} = \begin{bmatrix} \mathbf{u}_1 \ldots & \mathbf{u}_p\mathbf{u}_{p+1} \ldots & \mathbf{u}_m  \end{bmatrix} \in \mathbb{R}^{m \times m}
$$

is an orthogonal matrix.

It remains to verify that $\mathbf{AV} = \mathbf{U \Sigma}$, i.e., 

$$  
\mathbf{A} \begin{bmatrix} \mathbf{v}_1 \ldots & \mathbf{v}_p ~ \mathbf{v}_{p+1} \ldots & \mathbf{v}_n\end{bmatrix} = \begin{bmatrix} \mathbf{u}_1 \ldots & \mathbf{u}_p ~ \mathbf{u}_{p+1} \ldots & \mathbf{u}_m\end{bmatrix} ~ \begin{bmatrix} \mathbf{\sigma}_1  &  & & &\\  & \ddots & & &\\ & & \sigma_p & & & \\ & & & & & \\ & & & & & \\  \end{bmatrix} 
$$

Consider two cases:
- $1\leq i \leq p$: $\mathbf{Av}_i = \sigma_i \mathbf{u}_i$ by construction.
- $i > p$: $\mathbf{Av}_i = \mathbf{0}$, which is due to $\mathbf{A}^T \mathbf{Av}_i = \mathbf{Bv}_i = 0 \mathbf{v}_i = \mathbf{0}$.


Consequently, we have obtained that $\color{yellow}\mathbf{A} = \mathbf{U \Sigma V}^T$.


> **<font color='magenta'>Lemma 1.</font>** _Matrices_ $A$ _and_ $B$ _are identical if and only if for all vectors_ $\mathbf{v}$, $A\mathbf{v}= B \mathbf{v}$ 

> **<font color='magenta'>Proof:</font>** Clearly, if $A=B$ then $A\mathbf{v} = B \mathbf{v}$ for all $\mathbf{v}$. For the converse, suppose that $A\mathbf{v} = B \mathbf{v}$ for all $\mathbf{v}$. Let $\mathbf{e_i}$ be the vector that is all zeros except for the $i^{th}$ component which has value 1. Now $A \mathbf{e_i}$ is the $i^{th}$ column of $A$ and thus $A=B$ for each $i$, $A\mathbf{e_i}= B\mathbf{e_i}$.

>**<font color='magenta'>Theorem 1.</font>** Let $A$ be an $m \times n$ matrix with right-singular vectors $\mathbf{v_1, v_2, \ldots, v_p}$, left-singular vectors $\mathbf{u_1, u_2, \ldots, u_p}$, and corresponding singular values $\sigma_1, \sigma_2, \ldots, \sigma_p$. Then

$$
A = \sum_{i=1}^p \sigma_i \mathbf{u_i v_i^T}
$$

>**<font color='magenta'>Proof:</font>** For each vector $\mathbf{v_j}$, $A\mathbf{v_j} = \sum_{i=1}^p \sigma_i \mathbf{u_i v_i^T v_j}$. Since any vector $\mathbf{v}$ can be expressed as a linear combination of the singular vector perpendicular to the $\mathbf{v_i}$, $A\mathbf{v} = \sum_{i=1}^p \sigma_i \mathbf{u_i v_i^T v}$ and by identicality of matrices $A$ and $B$ for all vectors $\mathbf{v}$, $A\mathbf{v} = B \mathbf{v}$ now we have $A = \sum_{i=1}^p \sigma_i \mathbf{u_i v_i^T}$.



### Example

The example below defines a $4 \times 2$ matrix and calculates the SVD.

```ts
# SVD
import numpy
from numpy import array
from scipy.linalg import svd
# define a matrix
A = array([[1, 2], [3, 4], [5, 6], [7, 8]])
print(A)
```

```ts
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
```

