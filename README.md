# EasyDistillation

EasyDistillation 是一个实用程序集，用于计算格点QCD中蒸馏方法框架下的中间数据，包括 Laplace 算符的本征向量，元算符（elemental）和馏分传播子（perambulator）。同时，它可以生成特定量子数的介子算符，并计算这些算符之间的关联函数。

## 本征向量

参考 `gen_evecs.py`，可以计算 Laplace 算符的本征向量。需要使用 [PyQuda](https://github.com/IHEP-LQCD/PyQuda) 项目进行规范涂摩（gauge smearing）。由于使用了 CuPy 进行本征系统求解，效率得到的极大提高。但是由于 CuPy 的 `cupyx.scipy.sparse.linalg.eigsh` 函数存在Bug，例如[这个](https://github.com/cupy/cupy/issues/7168)，以及有些算法尚未实现，需要略微修改 Laplace 算符。

大体上，只需要使用

```python
calc_laplace_eigs(gauge_path: str, evecs_path: str, nstep: int, rho: float, num_evecs: int, tol: float):
```

就可以从 `gauge_path` 读入一个 ILDG 格式的组态，并输出到 `evecs_path`，为 `npy` 格式，是 NumPy 数组的默认存储格式，存储顺序为 `[Ne][Lt][Lz * Ly * Lx * Nc]`。

## 元算符（Elemental）

参考 `gen_elemental.py`，可以读入上一节产生的本征向量并计算元算符。程序实现了向前向后导数 $\overleftrightarrow{\nabla}$，并可以传入动量模式列表。

输出的元算符数据为 `npy` 格式，存储顺序为 `[Nderiv][Nmom][Lt][Ne][Ne]`。`Nderiv` 为计算的导数数量，从 `0` 到 `Nderiv-1` 指标为 $\mathbb{I}$, $\overleftrightarrow{\nabla}_x$, $\overleftrightarrow{\nabla}_y$, $\overleftrightarrow{\nabla}_z$, $\overleftrightarrow{\nabla}_x\overleftrightarrow{\nabla}_x$, $\overleftrightarrow{\nabla}_y\overleftrightarrow{\nabla}_x$, $\overleftrightarrow{\nabla}_z\overleftrightarrow{\nabla}_x$, $\overleftrightarrow{\nabla}_x\overleftrightarrow{\nabla}_y$, ...，以此类推。实际上作为输入参数的是导数的最高阶数。

## 馏分传播子（Perambulator）

TODO: 需要 PyQuda 求逆。已经实现，等待添加。

## 介子两点关联函数

参考 `gen_twopt.py`，可以产生特定量子数的介子算符[1]，通过对它们线性组合（例如求解 GEVP 后的优化算符）后，读入元算符和馏分传播子，计算两点关联函数。主要可以查看 `twopoint`，`twopoint_matrix`，`twopoint_isoscalar`。

正在添加两粒子算符的构造和相应关联函数计算，计划添加重子算符构造。
