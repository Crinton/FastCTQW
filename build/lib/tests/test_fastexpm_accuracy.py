# test_fixture_scopes.py
import pytest
import numpy as np
import networkx as nx
from FastCTQW.fastexpm import ExpMatFloat32, ExpMatFloat64, ExpMatComplex64, ExpMatComplex128
from scipy.linalg import expm

np.random.seed(0)

def matrix_exp_taylor(A: np.ndarray, m: int = 20, tol: float = 1e-10) -> np.ndarray:
    """
    使用泰勒级数结合缩放和平方法计算矩阵指数 exp(A)`。

    泰勒级数：exp(A) ≈ I + A + A^2/2! + ... + A^m/m!
    缩放和平方法：先计算 exp(A/2^s)，然后通过 s 次平方得到 exp(A)`。

    参数:
        A: 输入的方阵，类型为 numpy.ndarray。
        m: 泰勒级数的最大阶数（默认 20）。
        tol: 收敛容差，当前项范数小于此值时停止迭代（默认 1e-10）。

    返回:
        expA: 计算得到的 exp(A) 的近似值。
    """
    assert A.shape[0] == A.shape[1], "矩阵 A 必须为方阵"
    n = A.shape[0]

    # 计算矩阵范数并确定缩放因子 s
    norm_A = np.linalg.norm(A, ord=1)  # 使用 1-范数
    if norm_A == 0:
        return np.eye(n, dtype=A.dtype)

    # 确定缩放因子 s，使得 A/2^s 的范数较小
    s = max(0, int(np.ceil(np.log2(norm_A))))  # 确保范数归一化到 [0,1]
    scaling_factor = 2 ** s
    # 缩放矩阵
    A_scaled = A / scaling_factor

    # 初始化泰勒级数
    expA_scaled = np.eye(n, dtype=A.dtype)  # 单位矩阵 I
    term = np.eye(n, dtype=A.dtype)    # 当前项 A^k / k!

    # 计算泰勒级数展开
    for k in range(1, m + 1):
        term = term @ A_scaled / k  # 递推计算 A_scaled^k / k!
        expA_scaled += term
        # 检查收敛
        if np.linalg.norm(term, ord='fro') < tol:
            break

    # 通过 s 次平方恢复 exp(A)
    expA = expA_scaled.copy()
    for _ in range(s):
        expA = expA @ expA

    return expA

# def test_randMatrix_Accuracy(N: int, dtype, request):
#     if dtype is np.complex64:
#         A = np.asarray(1j*np.random.normal(size = (N,N)), dtype = np.complex64)
#         expMator = ExpMatComplex64(N)
#     elif dtype is np.complex128:
#         A = np.asarray(1j*np.random.normal(size = (N,N)), dtype = np.complex128)
#         expMator = ExpMatComplex128(N)

#     eA = matrix_exp_taylor(A,m = 1000)
#     my_eA = expMator.run(A).reshape(N,N)
#     error_matrix = eA - my_eA
#     l2_norm = np.linalg.norm(eA-my_eA)
#     l_inf_norm = np.linalg.norm(eA - my_eA, ord = np.inf)
#     max_norm = np.max(np.abs(error_matrix))
#     print(f"Random Matrix, N = {N}, dtype = {dtype}: ")
#     print(f"  L2 Norm (Frobenius): {l2_norm:.4e}")
#     print(f"  L_inf Norm (Max Row Sum): {l_inf_norm:.4e}")
#     print(f"  Max Norm (Element-wise): {max_norm:.4e}")
#     eA_l2_norm = np.linalg.norm(eA, ord='fro')
#     if eA_l2_norm > 1e-15:
#         relative_l2_error = l2_norm / eA_l2_norm
#         print(f"  Relative L2 Error: {relative_l2_error:.4e}")
#     else:
#         print("  Relative L2 Error: Cannot calculate (reference L2 norm is too small)")
    
def test_AdjMatComplete_Accuracy(N: int, dtype, request):
    if dtype is np.complex64:
        A = 1j*np.asarray(np.ones(shape = (N,N)) - np.eye(N), dtype = np.complex64)
        expMator = ExpMatComplex64(N)
    elif dtype is np.complex128:
        A = 1j*np.asarray(np.ones(shape = (N,N)) - np.eye(N), dtype = np.complex128)
        expMator = ExpMatComplex128(N)
    eA = matrix_exp_taylor(A,m = 100)
    my_eA = expMator.run(A).reshape(N,N)
    error_matrix = eA - my_eA
    l2_norm = np.linalg.norm(eA-my_eA)
    l_inf_norm = np.linalg.norm(eA - my_eA, ord = np.inf)
    max_norm = np.max(np.abs(error_matrix))
    print(f"Adjacency Matrix of Complete N = {N}, dtype = {dtype}: ")
    print(f"  L2 Norm (Frobenius): {l2_norm:.4e}")
    print(f"  L_inf Norm (Max Row Sum): {l_inf_norm:.4e}")
    print(f"  Max Norm (Element-wise): {max_norm:.4e}")
    eA_l2_norm = np.linalg.norm(eA, ord='fro')
    if eA_l2_norm > 1e-15:
        relative_l2_error = l2_norm / eA_l2_norm
        print(f"  Relative L2 Error: {relative_l2_error:.4e}")
    else:
        print("  Relative L2 Error: Cannot calculate (reference L2 norm is too small)")
        
def test_AdjMatER_Accuracy(N: int, dtype, request):
    if dtype is np.complex64:
        G = nx.gnp_random_graph(n = N, p = 0.05)
        A = 1j*nx.to_numpy_array(G, dtype = np.complex64)
        expMator = ExpMatComplex64(N)
    elif dtype is np.complex128:
        G = nx.gnp_random_graph(n = N, p = 0.05)
        A = 1j*nx.to_numpy_array(G, dtype = np.complex128)
        expMator = ExpMatComplex128(N)
    eA = matrix_exp_taylor(A,m = 100)
    my_eA = expMator.run(A).reshape(N,N)
    error_matrix = eA - my_eA
    l2_norm = np.linalg.norm(eA-my_eA)
    l_inf_norm = np.linalg.norm(eA - my_eA, ord = np.inf)
    max_norm = np.max(np.abs(error_matrix))
    print(f"Adjacency Matrix of ER N = {N}, dtype = {dtype}: ")
    print(f"  L2 Norm (Frobenius): {l2_norm:.4e}")
    print(f"  L_inf Norm (Max Row Sum): {l_inf_norm:.4e}")
    print(f"  Max Norm (Element-wise): {max_norm:.4e}")
    eA_l2_norm = np.linalg.norm(eA, ord='fro')
    if eA_l2_norm > 1e-15:
        relative_l2_error = l2_norm / eA_l2_norm
        print(f"  Relative L2 Error: {relative_l2_error:.4e}")
    else:
        print("  Relative L2 Error: Cannot calculate (reference L2 norm is too small)")