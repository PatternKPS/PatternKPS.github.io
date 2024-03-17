---
layout:     post
title:      "Singular Value Decomposition (SVD)"
date:       2024-03-15 15:58:00
author:     "Kristina"
type: "text/x-mathjax-config"
tags:
    - svd
---

### Prelimenary


Let bold-face lower-case letters (like $`\mathbf{a}`$) refer to <font color='yellow'>vectors</font>, bold-face capital letters $\mathbf{A}$ refer to <font color='yellow'>matrices</font>, and italic lower-case letters (like $\mathcal{a}$) refer to <font color='yellow'>scalars</font>. Think $\mathbf{A}$ is a  square or rectangular. Its rank is $p$. We will diagonalize this $A$, but not by $X^{-1} AX$. The *eigenvectors* in $X$ have three big problems:
1. They are usually not orthogonal
2. There are not always enough *eigenvectors*, and
3. $Ax = \lambda x$ requires $A$ to be a square matrix

The **<font color='yellow'>singular vectors</font>** of $A$ solve all those three problems in a perfect way.  

### Defenition

1. The SVD of a matrix is a sort of change of coordinates that makes the matrix simple, a *generalization of diagonalization*. It has some interesting algebraic properties and conveys important geometrical and theoritical insights about **linear transformations**.

2. The **singular value decomposition (SVD)** of a matrix is a **factorization** of that matrix **into three matrices**, where the factorization has the form $\color{yellow}\mathbf{U\Sigma V}$. 

3. A **<font color='yellow'>factorization</font>** is called the **<font color='yellow'>eigendecomposition</font>** of A, also called
the **<font color='yellow'>spectral decomposition</font>** of $\mathbf{A}$.

4. $\mathbf{U}$ is an $m \times p$ matrix, $\mathbf{\Sigma}$ is a $p\times p$ diagonal matrix, and $\mathbf{V}$ is an $n \times p$ matrix, with $\mathbf{V}^T$ being the transpose of $V$, a $p \times n$ matrix. The value $p$ is called the **<font color='green'>rank</font>**. 

5. The diagonal entries of $\mathbf{\Sigma}$ are referred to as _the singular values of_ $\mathbf{A}$. The columns of $\mathbf{U}$ are typically called *the left-singular vectors* of $\mathbf{A}$, and the columns of $V$ are called *the right-singular vectors of* $\mathbf{A}$. 

6. Unfortunately not all matrices can be diagonalized. **<font color='green'>SVD is a way to do something like diagonalization for any matrix, even non-square matrices</font>**.

