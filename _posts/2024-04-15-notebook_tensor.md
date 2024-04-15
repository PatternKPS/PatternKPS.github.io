---
layout:     post
title:      "Tensor"
date:       2024-04-15 09:05:00
author:     "Kristina"
header-style: text
hidden:       false
catalog:      true
mathjax:      true
tags:
    - Tensor
---

### Tensor: Applied Mathematics


The era of AI (soon will become AGI), big data, and IoT is just getting started. These three concepts are related to each other and becomes important to drive major innovations in digital transformation. The big data is a core subset of the IoT, and AI algorithms powered its ability to provide and extract information. 

The future of technology lies in data and data analysis. Given simple examples, doctors defining categories of tumours by their properties, astronomers grouping galaxies by their shape, achaelogists defining cultural periods from features of found artifacts, programs that label the pixels of an image by the object they belong to, other programs that segment a video stream into scenes, recommender systems that classification and categorisation are very basic tasks for the development of human language and conceptual thinking (Inspired by <a href="http://www.homepages.ucl.ac.uk/~ucakche/presentations/g19lecnotes.pdf">Dr Christian Hennig'lecture notes</a>). All these tasks nowadays are supported by the use of algorithms. 

In the past few decades, the algebraic structure of tensor has promoted the development of many fields, including *<font color='black'>signal processing, image processing, machine learning, numerical calculation, quantum physics, quantum chemistry, neuroscience, communication, psychometrics, chemometrics, biometrics etc</font>*. 

In recent years, the **<font color='black'>technology of tensor computing</font>**  has **<font color='black'>attracted</font>** researchers from different field of areas. The reasons are: **<font color='black'>first</font>**, real-world data are often tensors in nature; **<font color='black'>second</font>**, the parameters of many data-driven models have tensor structures. Therefore, the concept of tensor computing is become a core thing to address *data dimensionality reduction*, *pattern recognition*, *data fusion*, *multidimensional signal reconstruction*, *time series analysis*, *image restoration*, etc. 

**<font color='black'>A tensor</font>** is a higher-order arrays that represents a specific kind of data. *In general sense, a tensor is a multi-dimensional array*. Compared with a *<font color='black'>two-dimensional array (matrix)</font>*, *<font color='black'>a tensor has a more complex and flexible algebraic structure</font>*. The concept of tensors plays a crucial role in the development of algorithms for data analysis, particularly in the context of high-dimensional and structured data. Tensors provide a powerful mathematical framework for representing and analyzing complex data relationships, allowing algorithms to capture and exploit the inherent multidimensional structures in the data.

Tensor-based algorithms, such as tensor factorization and decomposition techniques (e.g., <font color='black'>PARAFAC/ Canonical Polyadic (CP)</font>, <font color='black'>Tucker decompositions</font> and <font color='black'>Non-Negative Tensor Factorization (NTF)</font>), simultaneosly provide powerful tools for dimensionality reduction and enable the analysis of components from tensors. These algorithms can extract low-rank approximations or latent factors from high-dimensional tensors, effectively reducing the dimensionality of the data while preserving its essential characteristics. <font color='black'>Dimensionality reduction</font> *enables* efficient storage, computation, and analysis of large-scale data, facilitating more scalable and effective algorithms. For example, <font color='black'>tensor-based clustering algorithms</font> can uncover clusters that exhibit correlations in multiple atributes, leading to more comprehensive and accurate clustering results. By considering the <font color='black'>multilinear relationships</font>, algorithms can uncover hidden insights and dependencies that may be missed by traditional data analysis techniques. 

The use of tensor concepts in algorithm development for data analysis facilitates the representation, analysis, and interpretation of complex, high-dimensional data. Tensors enable algorithms to capture and leverage the intrinsic structures and relationships within the data, leading to more efficient, accurate, and scalable solutions in <font color='black'>the era of data-driven technologies</font>.


### Tensor: Multilinear Algebra

Tensor algebra is a branch of mathematics that deals with the properties and manipulations of tensors. Tensors are mathematical objects used to represent multilinear relationships between *vector spaces. They generalize scalars, vectors, and matrices to higher dimensions. In fact, vectors and matrices are special cases of tensors, which can be defined as low-level tensors.


In Multilinear algebra, we work with *vector spaces* and their dual spaces. A vector space is a collection of objects called vectors, which can be added together and scaled by *scalars*. The dual space* of a vector consists of linear functionals, which are mappings from the vector space to the field of scalars *(usually real numbers or complex numbers).


A tensor can be described as a multidimensional or N-way array. A tensor can be called a generalized matrix. A tensor is identified by three parameters, such as rank, shape, and size. Rank is referring to the number of tensor's dimensions. Shape is referring to the number of tensor's columns and rows. CANDECOMP/PARAFAC (CP) decomposes a tensor as *a sum of rank-one tensors. The Tucker decomposition is a higher-order form of principal component analysis (PCA). There are many other tensor decompositions, including INdividual Differences in multidimensional SCALing (INDSCAL), Parallel factor analysis 2 (PARAFAC2), Canonical Decomposition with Linear Constraints (CANDELINC), DEcomposition into DIrectional COMponents (DEDICOM), and PARATUCK2. Tensor factorization is used to extract latent features that can facilitate discoveries of new mechanisims and signatures hidden in the data, where the explainability of the latent features is of principal importance.


**Notation**

1. *Order* is the number of ways or modes of a tensor. 

2. *Vectors* **(1D Matrix)** are tensors of order one and denoted by boldface lowercase letters, e.g. $\mathbf{t}$.

3. *Matrices* are tensors of order two and denoted by boldface capital letters, e.g. $\mathbf{T}$. 

4. Tensors of higher-order, namely order three and greater, we denote by boldface Euler script letters, e.g. $\mathbf{\mathcal{T}}$.

5. Thus, if $\mathbf{\mathcal{T}}$ represent a $D$ way data array of size $n_1 \times n_2 \times \ldots \times n_D$, we say $\mathbf{\mathcal{T}}$ is a tensor of order $D$. 

6. Denote scalars by lowercase letters, e.g. $t$. 

7. Denote the $i$- th element of a vector $\mathbf{t}$ by $t_i$, the $ij$- th element of a matrix $\mathbf{T}$ by $t_{ij}$, the $ijk$- th element of a third-order tensor $\mathbf{\mathcal{T}}$ by $t_{ijk}$, and so on.

8. We denote the $i$-th row of a matrix $\mathbf{T}$ by $\mathbf{T_{i:}}$ and the $j$-th column of a matrix $\mathbf{T}$ by $\mathbf{T_{:j}}$.

9. *Fibers* are subarrays of a tensor obtained by fixing all but one of its indices. In the case of a matrix, a *mode-1 fiber* is a *matrix column* and a *mode-2 fiber* is a *matrix row*.

10. *Slices* are the two-dimensional subarrays of a tensor obtained by fixing all but two indices. For example, a third-order tensor $\mathbf{\mathcal{T}}$ has three sets of slices denoted by $\mathbf{\mathcal{T_{i,::}}}$, $\mathbf{\mathcal{T_{:j:}}}$, and $\mathbf{\mathcal{T_{::k}}}$.


It is often convenient to reorder the elements of a $D$- way array into a matrix or vector. 

- Reordering a tensor's elements into a matrix is reffered to as *matricization*, while
- Reordering its elements into a vector is referred to as *vectorization*. 


There are many ways to reorder a tensor into a matrix or vector. We can use :

1. Canonical mode-$d$ matricization or well-known as PARAFAC/ Canonical Polyadic (CP),
2. Tucker decompositions, and 
3. Non-Negative Tensor Factorization (NTF)
