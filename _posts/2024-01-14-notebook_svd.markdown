---
layout:     post
title:      "Singular Value Decomposition (SVD)"
date:       2024-03-15 15:45:00
author:     "Kristina"
header-img: ""
tags:
    - U-MV-FCM
    - Multi-view clustering
    - MV-FCM
    - Optimal c
    - My notebook
---

<div class="content">
## Prelimenary


Let bold-face lower-case letters (like $\color{yellow}\mathbf{a}$) refer to <font color='yellow'>_vectors_</font>, bold-face capital letters $\color{yellow}\mathbf{A}$ refer to <font color='yellow'>_matrices_</font>, and italic lower-case letters (like $\color{yellow}\mathcal{a}$) refer to <font color='yellow'>_scalars_</font>. Think $A$ is a $m \times n$ matrix, square or rectangular. Its rank is $p$. We will diagonalize this $A$, but not by $X^{-1} AX$. The _eigenvectors_ in $X$ have three big problems:
1. They are usually not orthogonal
2. There are not always enough _eigenvectors_, and
3. $Ax = \lambda x$ requires $A$ to be a square matrix

The <font color='yellow'>**_singular vectors_**</font> of $A$ solve all those three problems in a perfect way.  

### Defenition

1. The SVD of a matrix is a sort of change of coordinates that makes the matrix simple, a _generalization of diagonalization_. It has some interesting algebraic properties and conveys important geometrical and theoritical insights about **linear transformations**.

2. The **singular value decomposition (SVD)** of a matrix is a **factorization** of that matrix **into three matrices**, where the factorization has the form $\color{yellow}\mathbf{U\Sigma V}$. 

3. A **<font color='yellow'>factorization</font>** is called the **<font color='yellow'>eigendecomposition</font>** of A, also called
the **<font color='yellow'>spectral decomposition</font>** of $\mathbf{A}$.

4. $\mathbf{U}$ is an $m \times p$ matrix, $\mathbf{\Sigma}$ is a $p\times p$ diagonal matrix, and $\mathbf{V}$ is an $n \times p$ matrix, with $\mathbf{V}^T$ being the transpose of $V$, a $p \times n$ matrix. The value $p$ is called the **<font color='green'>rank</font>**. 

5. The diagonal entries of $\mathbf{\Sigma}$ are referred to as _the singular values of_ $\mathbf{A}$. The columns of $\mathbf{U}$ are typically called _the left-singular vectors_ of $\mathbf{A}$, and the columns of $V$ are called _the right-singular vectors of_ $\mathbf{A}$. 

6. Unfortunately not all matrices can be diagonalized. **<font color='green'>SVD is a way to do something like diagonalization for any matrix, even non-square matrices</font>**.

## Application

1. In many applications, the data matrix $A$ is close to a matrix of low rank and it is useful to find a low rank matrix which is a good approximation to the data matrix . We will show that from the singular value decomposition of $A$, we can **<font color='green'>get the matrix</font>** $B$ **<font color='green'>of rank</font>** $p$ **<font color='green'>which best approximates </font>**$A$.

2. Principal component analysis (PCA)

3. Clustering a mixture of spherical Gaussians

4. To solve discrete optimization problem

5. For spectral decomposition

6. It can be used to **<font color='green'>reduce the dimensionality </font>**, i.e., the number of columns, of a dataset. 

7. It is often used in **<font color='green'>digital signal processing</font>** for **<font color='magenta'>noise reduction</font>**, **<font color='magenta'>image compression</font>**, and other areas.

### Mathematics behind SVD

**<font color='green'>Theorem 1.</font>** For any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, there exist two orthogonal matrices $\mathbf{U} \in \mathbb{R}^{m \times m}$, $\mathbf{V} \in \mathbb {R}^{n \times n}$, and a non-negative "diagonal" matrix $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$ (of the same size as $\mathbf{A}$). Such that the SVD of $m \times n$ matrix $\mathbf{A}$ is given by the formula:

\begin{equation*}
\color{yellow}\mathbf{A}_{m\times n} = \mathbf{U}_{m \times m} \mathbf{\Sigma}_{m \times n} \mathbf{V}_{n \times n}^T
\tag{1}
\end{equation*}

where:
- $\color{yellow} \mathbf{U}$: $m \times m$ matrix of the orthonormal _eigenvectors_ of $\mathbf{AA}^T$.
- $\color{yellow}\mathbf{\Sigma}$: an $m \times n$ matrix whose $i^{th}$ diagonal entry equals the $i^{th}$ singular value $\sigma_i$ for $i=1, 2, \ldots, p$. All other entries of $\mathbf{\Sigma}$ are zero.
- $\color{yellow}\mathbf{V}^T$: transpose of a $n \times n$ matrix containing the orthonormal eigenvectors of $\mathbf{A}^T \mathbf{A}$.

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
$$\mathbf{AV} = \mathbf{U\Sigma}$$
or in columns,
$$\mathbf{A} \begin{bmatrix} \mathbf{v}_1 \ldots & \mathbf{v}_p ~ \mathbf{v}_{p+1} \ldots & \mathbf{v}_n\end{bmatrix} = \begin{bmatrix} \mathbf{u}_1 \ldots & \mathbf{u}_p ~ \mathbf{u}_{p+1} \ldots & \mathbf{u}_m\end{bmatrix} ~ \begin{bmatrix} \mathbf{\sigma}_1  &  & & &\\  & \ddots & & &\\ & & \sigma_p & & & \\ & & & & & \\ & & & & & \\  \end{bmatrix}$$

By comparing columns, we obtain
\begin{equation*}
\mathbf{Av}_i=
\begin{cases}
\sigma_i \mathbf{u}_i,      &1<i \leq p ~~~~~~ \text{(non-zero singular values)}\\ 
0,      & p<i \leq n
\end{cases} \tag{3}
\end{equation*}

This tells us how to find the matrix $\mathbf{U} : \mathbf{u}_i = \frac{1}{\sigma_i} \mathbf{A} \mathbf{v}_i$ for $1\leq i \leq p$.

Let $\mathbf{B} = \mathbf{A}^T \mathbf{A} \in \mathbb{R}^{n \times n}$. Then $\mathbf{B}$ is square, symmetric, and positive semidefinite. 

Therefore, by the <font color='yellow'>_Spectral Theorem_</font>, $\mathbf{B} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$ for an orthogonal $\mathbf{V} \in \mathbb{R}^{n \times n}$ and diagonal $\mathbf{\Lambda} = \text{diag} (\lambda_1, \ldots, \lambda_n)$  with $\lambda_1 \geq \ldots \geq \lambda_p > 0 = \lambda_{p+1}=\ldots = \lambda_n$ (where $p = \text{rank} (\mathbf{A})\leq n)$.

Now let $\sigma_i = \sqrt{\lambda_i}$ and correspondingly from the matrix $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$:

$$\color{yellow} \mathbf{\Sigma} = \begin{bmatrix} \text{diag} ( \sigma_1, \ldots, \sigma_p) & \mathbf{O}_{p \times (n-p)} \\ \mathbf{O}_{(m-p) \times p} & \mathbf{O}_{(m-p)\times (n - p)} \end{bmatrix}$$

Define also

$$ \mathbf{u}_i = \frac{1}{\sigma_i} \mathbf{Av}_i \in \mathbb{R}^m, ~~~~~~~~~~~ \text{for each} ~~ 1\leq i\leq p$$


Then $\mathbf{u}_1, \ldots, \mathbf{u}_p$ are orthonormal vectors. To see this,

\begin{align*}
\mathbf{u}_i^T \mathbf{u}_j &= \bigg ( \frac{1}{\sigma_i} \mathbf{Av}_i \bigg)^T ~ \bigg ( \frac{1}{\sigma_j} \mathbf{Av}_j \bigg) & = \frac{1}{\sigma_i \sigma_j} \mathbf{v}_i^T ~ \underbrace{\mathbf{A}^T \mathbf{A}}_{\mathbf{B}} ~ \mathbf{v}_j &\\
 & = \frac{1}{\sigma_i \sigma_j} \mathbf{v}_i^T (\lambda_j \mathbf{v}_j) = \frac{\sigma_j}{\sigma_I} \mathbf{v}_i^T \mathbf{v}_j & &(\lambda_j = \sigma_j^2)\\
 & = \begin{cases}
1,      & i = j\\
0,      & i \neq j
\end{cases}
\end{align*}


Choose $\mathbf{u}_{p+1}, \ldots, \mathbf{u}_m \in \mathbb{R}^{m}$ (through basis completion) such that

$$\mathbf{U} = \begin{bmatrix} \mathbf{u}_1 \ldots & \mathbf{u}_p\mathbf{u}_{p+1} \ldots & \mathbf{u}_m  \end{bmatrix} \in \mathbb{R}^{m \times m}$$

is an orthogonal matrix.

It remains to verify that $\mathbf{AV} = \mathbf{U \Sigma}$, i.e., 

$$  \mathbf{A} \begin{bmatrix} \mathbf{v}_1 \ldots & \mathbf{v}_p ~ \mathbf{v}_{p+1} \ldots & \mathbf{v}_n\end{bmatrix} = \begin{bmatrix} \mathbf{u}_1 \ldots & \mathbf{u}_p ~ \mathbf{u}_{p+1} \ldots & \mathbf{u}_m\end{bmatrix} ~ \begin{bmatrix} \mathbf{\sigma}_1  &  & & &\\  & \ddots & & &\\ & & \sigma_p & & & \\ & & & & & \\ & & & & & \\  \end{bmatrix} $$

Consider two cases:
- $1\leq i \leq p$: $\mathbf{Av}_i = \sigma_i \mathbf{u}_i$ by construction.
- $i > p$: $\mathbf{Av}_i = \mathbf{0}$, which is due to $\mathbf{A}^T \mathbf{Av}_i = \mathbf{Bv}_i = 0 \mathbf{v}_i = \mathbf{0}$.

Consequently, we have obtained that $\color{yellow}\mathbf{A} = \mathbf{U \Sigma V}^T$.

> **<font color='magenta'>Lemma 1.</font>** _Matrices_ $A$ _and_ $B$ _are identical if and only if for all vectors_ $\mathbf{v}$, $A\mathbf{v}= B \mathbf{v}$ 

> **<font color='magenta'>Proof:</font>** Clearly, if $A=B$ then $A\mathbf{v} = B \mathbf{v}$ for all $\mathbf{v}$. For the converse, suppose that $A\mathbf{v} = B \mathbf{v}$ for all $\mathbf{v}$. Let $\mathbf{e_i}$ be the vector that is all zeros except for the $i^{th}$ component which has value 1. Now $A \mathbf{e_i}$ is the $i^{th}$ column of $A$ and thus $A=B$ for each $i$, $A\mathbf{e_i}= B\mathbf{e_i}$.

>**<font color='magenta'>Theorem 1.</font>** Let $A$ be an $m \times n$ matrix with right-singular vectors $\mathbf{v_1, v_2, \ldots, v_p}$, left-singular vectors $\mathbf{u_1, u_2, \ldots, u_p}$, and corresponding singular values $\sigma_1, \sigma_2, \ldots, \sigma_p$. Then

\begin{equation*}
A = \sum_{i=1}^p \sigma_i \mathbf{u_i v_i^T}
\end{equation*}

>**<font color='magenta'>Proof:</font>** For each vector $\mathbf{v_j}$, $A\mathbf{v_j} = \sum_{i=1}^p \sigma_i \mathbf{u_i v_i^T v_j}$. Since any vector $\mathbf{v}$ can be expressed as a linear combination of the singular vector perpendicular to the $\mathbf{v_i}$, $A\mathbf{v} = \sum_{i=1}^p \sigma_i \mathbf{u_i v_i^T v}$ and by identicality of matrices $A$ and $B$ for all vectors $\mathbf{v}$, $A\mathbf{v} = B \mathbf{v}$ now we have $A = \sum_{i=1}^p \sigma_i \mathbf{u_i v_i^T}$.

## Example

The example below defines a $4 \times 2$ matrix and calculates the SVD.


```python
# SVD
import numpy
from numpy import array
from scipy.linalg import svd
# define a matrix
A = array([[1, 2], [3, 4], [5, 6], [7, 8]])
print(A)

```

    [[1 2]
     [3 4]
     [5 6]
     [7 8]]


Find the singular value decomposition of a rectangular matrix A


```python
# SVD
U, s, VT = svd(A)
```

Running the example 


```python
print(U)
```

    [[-0.15248323 -0.82264747 -0.39450102 -0.37995913]
     [-0.34991837 -0.42137529  0.24279655  0.80065588]
     [-0.54735351 -0.0201031   0.69790998 -0.46143436]
     [-0.74478865  0.38116908 -0.5462055   0.04073761]]



```python
print(s)
```

    [14.2690955   0.62682823]



```python
print(VT)
```

    [[-0.64142303 -0.7671874 ]
     [ 0.7671874  -0.64142303]]



```python
# Create n x n Sigma matrix

from numpy import zeros
# create m x n Sigma matrix
Sigma = zeros((A.shape[0], A.shape[1]))
# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[1], :A.shape[1]]=diag(s)
print(Sigma)
```

    [[14.2690955   0.        ]
     [ 0.          0.62682823]
     [ 0.          0.        ]
     [ 0.          0.        ]]


Confirm the relation $\color{yellow} \mathbf{A = U \Sigma V}^T$, within machine precision 


```python
# reconstruct matrix
from numpy import dot
B1=U.dot(Sigma.dot(VT))
print(B1)
```

    [[1. 2.]
     [3. 4.]
     [5. 6.]
     [7. 8.]]


As can be seen above, matrix B1 (the original matrix A) reconstructed from the SVD elements.

## Pseudo-inverse of an arbitrary matrix

#### **<font color='green'>Why should we care about the concept of the pseudo-inverse of a matrix??</font>** &#129300; &#129300; &#129300;


#### &#128073; **<font color='green'> For a linear equation system, we can compute the inverse of a square matrix to solve</font>** $x$ **<font color='green'>such as</font>** $Ax = b \Rightarrow x = A^{-1} b$. **<font color='green'>But not all matrices are invertible. Also, in _machine learning_ (ML), it will be unlikely to find an exact solution with the presence of _noise in data_. Our objective is to find the model that best fit the data. To do so, _Pseudo-inverse_ allows us to get some sort of a solution even if it is not a perfect solution.**

**Calculation of Pseudo-inverse:** **<font color='yellow'>Pseudo inverse</font>** or **<font color='yellow'>Moore-Penrose inverse</font>** is the generalization of the matrix inverse that may not be invertible (such as low-rank matrices). If the matrix invertible then its inverse will be equal to pseudo inverse but pseudo inverse exists for the matrix that is not invertible. 

The application of pseudo-inverse:

1. To compute a "best fit" (least squares) solution to a system of linear equations that lacks a solution.
2. To find the minimum (Euclidean) norm solution to a system of a linear equations with multiple solutions.
3. It facilitates the statement and proof of results in linear algebra. 


Let $\mathbf{A=U \Sigma V}^T$. The **<font color='yellow'>pseudo-inverse of a matrix</font>** $\color{yellow}\mathbf{A}$, denoted as $\color{yellow}A^{\dagger}$. 

> **<font color='magenta'>Definition</font>**:
For $\mathbf{A} \in \mathbb{R}^{m \times n}$, a pseudo-inverse of $\mathbf{A}$ is defined as a matrix $\mathbf{A}^{\dagger} \in \mathbb{R}^{n \times m}$ satisfying all of the following **<font color='green'>four criteria</font>**, known as **<font color='magenta'>the Moore-Penrose conditions</font>**. Hermitian matrix denoted as $\mathbf{A}^*$.

1. $\mathbf{AA}^{\dagger}$ need not be the general identity matrix, but it maps all column vectors of $\mathbf{A}$ to themselves:
   $$\mathbf{AA}^{\dagger} ~ \mathbf{A} = \mathbf{A}$$

2. $\mathbf{A}^{\dagger}$ acts like a _weak inverse_:
   $$\mathbf{A} ~ \mathbf{AA}^{\dagger} = \mathbf{A}^{\dagger}$$

3. $\mathbf{AA}^{\dagger}$ is _Hermitian_:
   $$(\mathbf{AA}^{\dagger})^* = \mathbf{AA}^{\dagger}$$

> **<font color='magenta'>Hermitian matrix</font>** is a special matrix; etymologically, it was named after a French mathematician Charles Hermite (1822-1901), who was trying to study the matrices that always have real _Eigenvalues_.


> $ \boxed{\text{A Hermitian} \Longleftrightarrow  a_{ij} = \overline{a_{ji}}}$

> $ \boxed{\text{A Hermitian} \Longleftrightarrow  \mathbf{A} = \overline{\mathbf{A}^{T}}}$

> $ \boxed{\text{A Hermitian} \Longleftrightarrow  \mathbf{A} = \mathbf{A}^*}$



4. $\mathbf{A}^{\dagger}~ \mathbf{A}$ is also Hermitian:
   $$(\mathbf{A}^{\dagger} ~ \mathbf{A})^* = \mathbf{A}^{\dagger} ~ \mathbf{A}$$

> Suppose, we need to calculate the pseudo-inverse of a matrix $\mathbf{A}$:
Then, the SVD of $\mathbf{A}$ can be given as:
\begin{equation*}
\mathbf{A = U \Sigma V}^T ~~~~~ \text{(multiply both sides by } \mathbf{A}^{-1})
\end{equation*}
then we have
\begin{align*}
\mathbf{A}^{-1}\mathbf{A} &= \mathbf{A}^{-1} \mathbf{U} \mathbf{\Sigma V}^T & \text{(knowing } \mathbf{A}^{-1}\mathbf{A} = \mathbf{I} )\\
\mathbf{I} &= \mathbf{A}^{-1} \mathbf{U} \mathbf{\Sigma V}^T  &\text{(multiply both sides by } \mathbf{V} )\\
\mathbf{V} &= \mathbf{A}^{-1} \mathbf{U \Sigma V}^T \mathbf{V} &\text{(multiply $\mathbf{\Sigma}$ by } \mathbf{\Sigma}^{-1} )\\
\mathbf{V \Sigma}^{-1}&= \mathbf{A}^{-1} \mathbf{U \Sigma\Sigma}^{-1} \mathbf{V}^T \mathbf{V} &\text{(multiply $\mathbf{U}$ by } \mathbf{U}^T )\\
\mathbf{V} \mathbf{\Sigma}^{-1}\mathbf{U}^T&= \mathbf{A}^{-1} \mathbf{UU}^T \mathbf{\Sigma\Sigma}^{-1}\mathbf{V}^T \mathbf{V} & \text{(knowing } \mathbf{UU}^T = \mathbf{I}, \mathbf{\Sigma\Sigma}^{-1}= \mathbf{I}, \mathbf{V}^T \mathbf{V}=\mathbf{I} )\\
\mathbf{V \Sigma}^{-1}\mathbf{U}^T&= \mathbf{A}^{-1}=\mathbf{A}^{\dagger}&
\end{align*}
The above equation gives the pseudo-inverse. 


## Conclusion

1. $ \color{yellow}\boxed{\text{SVD of  } \mathbf{A} \Longleftrightarrow  \mathbf{A} = \mathbf{U \Sigma V}^T}$ and 

2. $ \color{yellow}\boxed{\text{Pseudo-inverse of  } \mathbf{A} \Longleftrightarrow  \mathbf{A}^{\dagger} = \mathbf{V} \Sigma^{-1}\mathbf{U}^T}$


## Implementation

In this code, we will try to calculate the SVD using Numpy and Scipy. We will be calculating SVD, and also performing pseudo-inverse. In the end, we can apply SVD for compressing the image.


```python
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

    [[1 2]
     [3 4]
     [5 6]
     [7 8]]


Perform SVD


```python
U, s, VT = svd(A)
```

Print different components


```python
print("U: ", U)
print("Singular  array ", s)
print("V^{T}: ", VT)
```

    U:  [[-0.15248323 -0.82264747 -0.39450102 -0.37995913]
     [-0.34991837 -0.42137529  0.24279655  0.80065588]
     [-0.54735351 -0.0201031   0.69790998 -0.46143436]
     [-0.74478865  0.38116908 -0.5462055   0.04073761]]
    Singular  array  [14.2690955   0.62682823]
    V^{T}:  [[-0.64142303 -0.7671874 ]
     [ 0.7671874  -0.64142303]]


Calculate pseudo-inverse


```python
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

    Sigma is  [[0.07008153 0.        ]
     [0.         1.59533338]
     [0.         0.        ]
     [0.         0.        ]]
    The pseudo-inverse of matrix A is  [[-1.00000000e+00 -5.00000000e-01  1.61159432e-15  5.00000000e-01]
     [ 8.50000000e-01  4.50000000e-01  5.00000000e-02 -3.50000000e-01]]


#### SVD on Image Compression


```python
cat = data.chelsea()
plt.imshow(cat)
```




    <matplotlib.image.AxesImage at 0x11f571bd0>




    
![png](output_34_1.png)
    


Convert image to grayscale


```python
gray_cat = rgb2gray(cat)
plt.imshow(gray_cat)
```




    <matplotlib.image.AxesImage at 0x11f5f9b10>




    
![png](output_36_1.png)
    


Calculate the SVD and plot the image


```python
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

    U:  [[ 0.05169698  0.07354608  0.0441153  ...  0.07988875  0.0584584
       0.03824287]
     [ 0.0516497   0.07471783  0.04653544 ... -0.10084738 -0.14909323
      -0.05003661]
     [ 0.05163224  0.07531087  0.05024318 ...  0.02554351  0.18181389
       0.02204087]
     ...
     [ 0.06677288 -0.0311474   0.03493233 ... -0.00115441  0.00689057
      -0.04745911]
     [ 0.0668638  -0.02720109  0.03599754 ... -0.02132047 -0.00986954
       0.00101149]
     [ 0.06691356 -0.02633296  0.03607362 ... -0.00043176 -0.04172398
       0.00479383]]
    Singular  array  [[1.70427845e+02 0.00000000e+00 0.00000000e+00 ... 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 2.16412563e+01 0.00000000e+00 ... 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 1.73935477e+01 ... 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     ...
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 ... 2.29343946e-02
      0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 ... 0.00000000e+00
      2.28706468e-02 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 ... 0.00000000e+00
      0.00000000e+00 2.15307518e-02]]
    V^{T}:  [[ 0.04851108  0.04826917  0.04809735 ...  0.0513346   0.05129698
       0.05123909]
     [ 0.13872869  0.13975339  0.13977431 ... -0.07752516 -0.07770908
      -0.07779836]
     [ 0.03622185  0.03726394  0.03861645 ...  0.00426306  0.00440011
       0.00495034]
     ...
     [-0.0270361   0.00691155 -0.05534442 ... -0.06770688  0.00511687
      -0.14559302]
     [ 0.0514932  -0.03523579 -0.08264011 ... -0.05158923  0.03541566
       0.07250193]
     [ 0.00818938 -0.02067411 -0.00056798 ...  0.00528485 -0.02192733
       0.08231456]]



    
![png](output_38_1.png)
    

</div>
