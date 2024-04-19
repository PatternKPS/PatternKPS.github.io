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


### Implementation

In this code, I will try to calculate the SVD using Numpy and Scipy. I will  calculate SVD, and also perform pseudo-inverse. In the end, I will apply SVD for compressing the image.

```ts
# Imports

from skimage.color import rgb2gray
from skimage import data
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from scipy.linalg import svd

"""
Singular Value Decomposition (SVD)
"""

# define a matrix
A = array([[1, 2], [3, 4], [5, 6], [7, 8]])
print(A)
```

Perform SVD

```ts
U, s, VT = svd(A)
```

Print different components

```ts
print("U: ", U)
print("Singular  array ", s)
print("V^{T}: ", VT)
```

Calculate pseudo-inverse

```ts
from numpy import zeros
from numpy import diag
from numpy import dot

# create the reciprocal of s
d = 1.0/s
# create m x n matrix of zeroes 
D = zeros(A.shape)
# populate D with n x n diagonal matrix 
D[:A.shape[1], :A.shape[1]]=diag(d)

print("Sigma is ", D)

# calculate pseudo-inverse
B = VT.T.dot(D.T).dot(U.T)
print("The pseudo-inverse of matrix A is ", B)
```

#### SVD on Image Compression

```ts
cat = data.chelsea()
plt.imshow(cat)
```

![image](https://github.com/PatternKPS/patternkps.github.io/assets/150363044/24458152-a5bd-4ce2-9c61-f12d0bb22d22)

Convert image to grayscale

```ts
gray_cat = rgb2gray(cat)
plt.imshow(gray_cat)
```

![image](https://github.com/PatternKPS/patternkps.github.io/assets/150363044/d341a115-c9d5-46b4-94ee-7e9891fffee9)

Calculate the SVD and plot the image


```ts
U, s, VT = svd(gray_cat, full_matrices=False)
s = np.diag(s)
print("U: ", U)
print("Singular  array ", s)
print("V^{T}: ", VT)
fig, ax  = plt.subplots(5, 2, figsize=(8,20))

curr_fig = 0
for p in [5, 10, 70, 100, 200]:
    cat_approx = U[:, :p] @ s[0:p, :p] @ VT[:p, :]
    ax[curr_fig][0].imshow(cat_approx, cmap='gray')
    ax[curr_fig][0].set_title("k = "+str(p))
    ax[curr_fig, 0].axis('off')
    ax[curr_fig][1].set_title("Original Image")
    ax[curr_fig][1].imshow(gray_cat, cmap='gray')
    ax[curr_fig, 1].axis('off')
    curr_fig += 1
plt.show()
```


