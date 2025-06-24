# test_fastexpm_performance.py
import pytest
import numpy as np
import networkx as nx
from FastCTQW.fastexpm import ExpMatFloat32, ExpMatFloat64, ExpMatComplex64, ExpMatComplex128
from scipy.linalg import expm # <--- 重新导入 expm

np.random.seed(0)

# 辅助函数，用于生成测试矩阵，避免在每个测试函数中重复代码
def _generate_matrix(N: int, dtype: np.dtype, graph_type: str):
    """Generates a matrix based on graph type and dtype."""
    if graph_type == "complete":
        if dtype is np.complex64:
            A_real = np.ones(shape=(N, N), dtype=np.float32) - np.eye(N, dtype=np.float32)
            A_imag = np.random.rand(N, N).astype(np.float32)
            A = (A_real + 1j * A_imag).astype(np.complex64)
            exp_mator = ExpMatComplex64(N)
        elif dtype is np.complex128:
            A_real = np.ones(shape=(N, N), dtype=np.float64) - np.eye(N, dtype=np.float64)
            A_imag = np.random.rand(N, N).astype(np.float64)
            A = (A_real + 1j * A_imag).astype(np.complex128)
            exp_mator = ExpMatComplex128(N)
        else:
            pytest.skip(f"Unsupported dtype {dtype} for Complete Graph benchmark.")
    elif graph_type == "er":
        if N < 2:
            pytest.skip(f"N={N} is too small for ER graph benchmark.")
        G = nx.gnp_random_graph(n = N, p = 0.05)
        if dtype is np.complex64:
            A_real = nx.to_numpy_array(G, dtype = np.float32)
            A_imag = np.random.rand(N, N).astype(np.float32)
            A = (A_real + 1j * A_imag).astype(np.complex64)
            exp_mator = ExpMatComplex64(N)
        elif dtype is np.complex128:
            A_real = nx.to_numpy_array(G, dtype = np.float64)
            A_imag = np.random.rand(N, N).astype(np.float64)
            A = (A_real + 1j * A_imag).astype(np.complex128)
            exp_mator = ExpMatComplex128(N)
        else:
            pytest.skip(f"Unsupported dtype {dtype} for ER Graph benchmark.")
    elif graph_type == "r4":
        if N % 2 != 0 or N < 5:
            pytest.skip(f"N={N} is not suitable for 4-regular graph benchmark.")
        G = nx.random_regular_graph(d = 4, n = N)
        if dtype is np.complex64:
            A_real = nx.to_numpy_array(G, dtype = np.float32)
            A_imag = np.random.rand(N, N).astype(np.float32)
            A = (A_real + 1j * A_imag).astype(np.complex64)
            exp_mator = ExpMatComplex64(N)
        elif dtype is np.complex128:
            A_real = nx.to_numpy_array(G, dtype = np.float64)
            A_imag = np.random.rand(N, N).astype(np.float64)
            A = (A_real + 1j * A_imag).astype(np.complex128)
            exp_mator = ExpMatComplex128(N)
        else:
            pytest.skip(f"Unsupported dtype {dtype} for R4 Graph benchmark.")
    else:
        pytest.fail(f"Unknown graph type: {graph_type}")
    
    return A, exp_mator # 返回 exp_mator 实例

# --- Benchmarks for FastCTQW.fastexpm ---

@pytest.mark.benchmark(group="matrix_exp")
def test_fastexpm_AdjMatComplete(N: int, dtype, benchmark):
    """
    Benchmark FastCTQW.fastexpm for Complete Graph adjacency matrix.
    """
    A, expMator = _generate_matrix(N, dtype, "complete")
    
    benchmark(lambda: expMator.run(A).reshape(N,N))
    
    expMator.free()


@pytest.mark.benchmark(group="matrix_exp")
def test_fastexpm_AdjMatER(N: int, dtype, benchmark):
    """
    Benchmark FastCTQW.fastexpm for Erdős–Rényi Graph adjacency matrix.
    """
    A, expMator = _generate_matrix(N, dtype, "er")
    
    benchmark(lambda: expMator.run(A).reshape(N,N))
    
    expMator.free()


@pytest.mark.benchmark(group="matrix_exp")
def test_fastexpm_AdjMatR4(N: int, dtype, benchmark):
    """
    Benchmark FastCTQW.fastexpm for 4-regular Graph adjacency matrix.
    """
    A, expMator = _generate_matrix(N, dtype, "r4")
    
    benchmark(lambda: expMator.run(A).reshape(N,N))
    
    expMator.free()

# --- Benchmarks for scipy.linalg.expm ---

@pytest.mark.benchmark(group="matrix_exp")
def test_scipy_expm_AdjMatComplete(N: int, dtype, benchmark):
    """
    Benchmark scipy.linalg.expm for Complete Graph adjacency matrix.
    """
    # 注意：这里不需要 expMator 实例，因为它只用于 FastCTQW
    A, _ = _generate_matrix(N, dtype, "complete")
    
    # 将 scipy.linalg.expm 的调用包装在 benchmark 夹具中
    benchmark(lambda: expm(A))


@pytest.mark.benchmark(group="matrix_exp")
def test_scipy_expm_AdjMatER(N: int, dtype, benchmark):
    """
    Benchmark scipy.linalg.expm for Erdős–Rényi Graph adjacency matrix.
    """
    A, _ = _generate_matrix(N, dtype, "er")
    
    benchmark(lambda: expm(A))


@pytest.mark.benchmark(group="matrix_exp")
def test_scipy_expm_AdjMatR4(N: int, dtype, benchmark):
    """
    Benchmark scipy.linalg.expm for 4-regular Graph adjacency matrix.
    """
    A, _ = _generate_matrix(N, dtype, "r4")
    
    benchmark(lambda: expm(A))