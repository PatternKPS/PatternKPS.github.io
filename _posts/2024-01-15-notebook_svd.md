---
layout:     post
title:      "Singular Value Decomposition (SVD)"
date:       2024-03-15 15:58:00
author:     "Kristina"
header-style: text
hidden:       false
catalog:      true
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

